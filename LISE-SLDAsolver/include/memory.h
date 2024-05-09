#pragma once

template <typename T>
void ZeroMemory(T* ptr, int count)
{
    assert(ptr != nullptr && count > 0);
    std::memset(ptr, 0, count * sizeof(T));
}

template <typename T>
void FillMemory(T* ptr, int count, T fillValue)
{
    assert(ptr != nullptr && count > 0);
    std::fill(ptr, ptr + count, fillValue);
}

template <typename T>
void CopyMemory(T* ptr, int count, const T* fillArray)
{
    assert(ptr != nullptr && fillArray != nullptr && count > 0);
    std::memcpy(ptr, fillArray, count * sizeof(T));
}

template <typename T>
T* Allocate(int count)
{
    assert(count > 0);

    auto ptr = (T*)malloc(count * sizeof(T));
    assert(ptr != nullptr);
    return ptr;
}

template <typename T>
T* AllocateZeroed(int count)
{
    auto ptr = Allocate(count);
    ZeroMemory(ptr, count);
    return ptr;
}

template <typename T>
T* AllocateFilled(int count, T fillValue = {})
{
    auto ptr = Allocate(count);
    FillMemory(ptr, count, fillValue);
    return ptr;
}

template <typename T>
T* AllocateCopy(int count, const T* fillArray)
{
    auto ptr = Allocate(count);
    CopyMemory(ptr, count, fillArray);
    return ptr;
}

template <typename T, typename F>
T* AllocateInit(int count, F init)
{
    auto ptr = Allocate(count);
    for (int i = 0; i < count; ++i)
    {
        ptr[i] = init(i);
    }
    return ptr;
}

template <typename T>
void Free(T*& ptr)
{
    free(ptr);
    ptr = nullptr;
}