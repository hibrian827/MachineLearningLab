def binary_search(arr, trgt):
    length = len(arr)
    left = 0
    right = length
    while left < right:
        mid = (left + right) // 2
        val = arr[mid]
        if val < trgt:
            left = mid + 1
        elif val > trgt:
            right = mid
        else:
            return mid
    return -1  # Target not found (index of target)

class Sorter:
    def __init__(self, method):
        self.method = method

    @staticmethod
    def of(method):
        return Sorter(method)

    def sort(self, arr):
        if self.method == 'selection_sort':
            return self.selection_sort(arr)

        elif self.method == 'insertion_sort':
            return self.insertion_sort(arr)

        elif self.method == 'merge_sort':
            return self.merge_sort(arr)

        else:
            raise ValueError(f'Unknown method: {self.method}')

    def selection_sort(self, arr):
        for i in range(len(arr)):
            min_idx = i
            for j in range(i+1, len(arr)):
                if arr[min_idx] > arr[j]:
                    min_idx = j
            temp = arr[i]
            arr[i] = arr[min_idx]
            arr[min_idx] = temp
        return arr

    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            val = arr[i]
            for j in range(i - 1, -1, -1):
                if arr[j] < val:
                    arr[j + 1] = val
                    break
                arr[j + 1] = arr[j]
                if j == 0:
                    arr[0] = val
        return arr

    def merge_sort(self, arr):
        length = len(arr)
        i = 1
        while i < length:
            for j in range(0, length, i * 2):
                temp = []
                if j + i > length:
                    break
                arr1 = arr[j : j + i]
                end = j + 2 * i
                if j + 2 * i > length :
                    end = length
                arr2 = arr[j + i : end]
                i1 = 0
                i2 = 0
                while i1 < len(arr1) and i2 < len(arr2):
                    if arr1[i1] > arr2[i2]:
                        temp.append(arr2[i2])
                        i2 += 1
                    else:
                        temp.append(arr1[i1])
                        i1 += 1
                if i1 < len(arr1):
                    temp += arr1[i1:]
                if i2 < len(arr2):
                    temp += arr2[i2:]
                arr[j : end] = temp
            i *= 2
        return arr