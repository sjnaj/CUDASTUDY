#include "common/book.h"
#include "parallelprogrammingbook/include/hpc_helpers.hpp"
struct Lock
{
    int *mutex;
    Lock()
    {
        int state = 0;
        cudaMalloc((void **)&mutex, sizeof(int));
        CUERR
        cudaMemcpy(mutex, &state, sizeof(int), H2D);
        CUERR
    }
    ~Lock()
    {
        cudaFree(mutex);
    }
    __device__ void lock()
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
    }
    __device__ void unlock()
    {
        atomicExch(mutex, 0);
    }
};

struct Entry
{
    unsigned int key;
    void *value;
    Entry *next;
};

struct Table
{
    size_t count;
    Entry **entries;
    Entry *pool;
    Entry *firstFree;
};

__device__ __host__ size_t hash(unsigned int value, size_t count)
{
    return value % count;
}

void initialize_table(Table &table, int entries, int elements)
{
    table.count = entries;
    cudaMalloc((void **)&table.entries, entries * sizeof(Entry *));
    CUERR
    cudaMemset(table.entries, 0, entries * sizeof(Entry *));
    CUERR
    cudaMalloc((void **)&table.pool, elements * sizeof(Entry));
    CUERR
}

void free_table(Table &table)
{
    cudaFree(table.pool);
    cudaFree(table.entries);
}

#define SIZE (100 * 1024 * 1024)
#define ELEMENTS (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES 1024

void copy_table_to_host(Table &const table, Table &hostTable)
{
    hostTable.count = table.count;
    hostTable.entries = (Entry **)calloc(table.count, sizeof(Entry *));
    hostTable.pool = (Entry *)malloc(ELEMENTS * sizeof(Entry));

    cudaMemcpy(hostTable.entries, table.entries, table.count * sizeof(Entry *), D2H);
    CUERR
    cudaMemcpy(hostTable.pool, table.pool, ELEMENTS * sizeof(Entry), D2H);
    CUERR

    for (int i = 0; i < table.count; i++)
    {
        if (hostTable.entries[i] != NULL)
        {
            hostTable.entries[i] = (Entry *)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool); // old_offset+new_start=new_pointer
        }
    }
    for (int i = 0; i < ELEMENTS; i++)
    {
        if (hostTable.pool[i].next != NULL)
        {
            hostTable.pool[i].next = (Entry *)((size_t)hostTable.pool[i].next - (size_t)table.pool + (size_t)hostTable.pool);
        }
    }
}

void verify_table(Table &const dev_table)
{
    Table table;
    copy_table_to_host(dev_table, table);

    int count = 0;
    for (size_t i = 0; i < table.count; i++)
    {
        Entry *current = table.entries[i];
        while (current != NULL)
        {
            ++count;
            if (hash(current->key, table.count) != i)
            {
                printf("%d hashed to %ld,but was located at %ld\n", current->key, hash(current->key, table.count), i);
            }
            current = current->next;
        }
    }

    if (count != ELEMENTS)
    {
        printf("%d elements found in hashtable.Shoule be %ld\n", count, ELEMENTS);
    }
    else
    {
        printf("ALl %d elements found in hashtable\n", count);
    }

    free(table.pool);
    free(table.entries);
}

__global__ void add_to_table(unsigned int *keys, void **values, Table table, Lock *lock)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (tid < ELEMENTS)
    {
        unsigned int key = keys[tid];
        size_t hashValue = hash(key, table.count);
        for (int i = 0; i < 32; i++) // 32个线程（warp）某时刻只有一个能获取锁，利用循环使它们一定程度上有序访问，减少竞争锁带来的损耗
        {
            if ((tid % 32) == i)
            {
                Entry *location = &(table.pool[tid]);
                location->key = key;
                location->value = values[tid];
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                lock[hashValue].unlock();

            }
        }
        tid += stride;
    }
}

int main()
{
    unsigned int *buffer = (unsigned int *)big_random_block(SIZE);
    TIMERSTART(hash)
    CUERR
    unsigned int *dev_keys;
    void **dev_values;
    cudaMalloc((void **)&dev_keys, SIZE);
    CUERR
    cudaMalloc((void **)&dev_values, SIZE);
    CUERR
    cudaMemcpy(dev_keys, buffer, SIZE, H2D);
    CUERR

    Table table;
    initialize_table(table, HASH_ENTRIES, ELEMENTS);
    Lock lock[HASH_ENTRIES];
    Lock *dev_lock;
    cudaMalloc((void **)&dev_lock, HASH_ENTRIES * sizeof(Lock));
    CUERR
    cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(Lock), H2D);
    CUERR

    add_to_table<<<60, 256>>>(dev_keys, dev_values, table, dev_lock);
    CUERR

    // cudaDeviceSynchronize();
    // CUERR
    TIMERSTOP(hash)
    CUERR

    verify_table(table);

    free_table(table);
    cudaFree(dev_lock);
    cudaFree(dev_keys);
    cudaFree(dev_values);
    free(buffer);
}