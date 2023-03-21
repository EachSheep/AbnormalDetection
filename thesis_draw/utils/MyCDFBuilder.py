import numpy as np
import collections

class MyCDFBuilder():
    def __init__(self) -> None:
        pass

    @staticmethod
    def buildcdf_percentage(values, reverse=False, dtype = np.float32, show_type = False):
        """
        横轴是排名，纵轴是个数
            show_type == True时，展示连续的数字比如有三个3,[2, 3, 3, 3], True时候的和为[2+3, 2+3+3, 2+3+3+3]
            show_type == False时，展示非连续的数字比如有三个3,[2, 3, 3, 3], True时候的和为[2, 2, 2, 2+3*3]
        """
        if not show_type:
            # 计算每一个值出现的次数，存储在values_list = [[], [], ...]中，values_list的每一个元素为 (元素值, 元素值出现的个数)的元组
            values_count = collections.Counter(values)
            values_list = [list(item) for item in values_count.items()]
            values_list.sort(reverse=reverse)
            values_list = np.array(values_list, dtype=dtype)

            cum_values_list = values_list[:, 0] * values_list[:, 1]
            cum_values_list = np.cumsum(cum_values_list).astype(dtype)
            
            new_values_list = [] # 每个元素是一个二维元组，(value1, value2)，value1代表横坐标，value2代表纵坐标
            k = 1
            for i in range(len(values_list)):
                num = int(values_list[i][1])
                for j in range(num):
                    new_values_list.append((k, cum_values_list[i]))
                    k += 1
            new_values_list = np.array(new_values_list, dtype=dtype)
            return new_values_list
        else:
            values = sorted(values, reverse = reverse)
            len_values = len(values)
            cum_values_list = np.cumsum(values).astype(dtype).reshape(len(values), 1)
            x = np.arange(1, len_values + 1, 1).reshape(len_values, 1)
            y = cum_values_list
            return np.concatenate((x, y), axis=1)

    @staticmethod
    def buildcdf_frequency(values, reverse=False, dtype = np.float32):
        """
        横轴是values里面的数字，纵轴是某个values里面的数字出现的个数
        """
        values_count = collections.Counter(values)
        values_list = [list(item) for item in values_count.items()]
        values_list.sort(reverse=reverse)
        values_list = np.array(values_list, dtype=dtype)
        
        values_list[:, 1] = np.cumsum(values_list[:, 1])
        return values_list
    

    @staticmethod
    def buildcdf_both_frequency(values, reverse=False, dtype = np.float32):
        """
        横轴是values里面的数字，纵轴是某个values里面的数字出现的个数
        纵坐标是值的个数而不是累加值
        """
        values_count = collections.Counter(values)
        values_list = [list(item) for item in values_count.items()]
        values_list.sort(reverse=reverse)
        values_list = np.array(values_list, dtype=dtype)
        return values_list

if __name__ == "__main__":
    pass
