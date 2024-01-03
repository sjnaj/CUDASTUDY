#include "common/book.h"
#include "cuda.h"

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

void initialize_table(Table &table, int entries, int elements)
{
    table.count = entries;
    table.entries = (Entry **)calloc(entries, sizeof(Entry *));
    table.pool = (Entry *)malloc(elements * sizeof(Entry));
    table.firstFree = table.pool;
}

void free_table(Table &table)
{
    free(table.entries);
    free(table.pool);
}

size_t hash(unsigned int key, size_t count)
{
    return key % count;
}

void add_to_table(Table &table, unsigned int key, void *value)
{
    size_t hashValue = hash(key, table.count);

    Entry *location = table.firstFree++;
    location->key = key;
    location->value = value;

    location->next = table.entries[hashValue];
    table.entries[hashValue] = location;
}

#define SIZE (100 * 1024 * 1024)
#define ELEMENTS (SIZE / sizeof(unsigned int))

void verify_table(Table &const table)
{
    int count = 0;
    for (size_t i = 0; i < table.count; i++)
    {
        Entry *current = table.entries[i];
        while (current != NULL)
        {
            count++;
            if (hash(current->key, table.count) != i)
            {
                printf("%d hashed to %ld,but was located at %ld\n", (unsigned int)current->value, hash(current->key, table.count), i);
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
}

#define HASH_ENTRIES 1024

int main()
{
    unsigned int *buffer = (unsigned int *)big_random_block(SIZE);
    clock_t start, stop;
    start = clock();
    Table table;
    initialize_table(table, HASH_ENTRIES, ELEMENTS);

    for (int i = 0; i < ELEMENTS; i++)
    {
        add_to_table(table, buffer[i], (void *)NULL);
    }
    stop = clock();
    float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("Time to hash: %3.1f ms\n", elapsedTime);

    verify_table(table);
    free_table(table);
    free(buffer);
    return 0;
}