import os


class GetAllFilesInPath:

    def __init__(self, dirPath=""):
        super(GetAllFilesInPath, self).__init__()
        self.dirPath = dirPath

    @staticmethod
    def getAllFileNamesInPath(dirPath: str) -> list:
        """
        input: 
            dirPath -> str: directory you want to get files from
        return: 
            list: list of paths of all files in dirPath
        """
        absDirPath = os.path.abspath(dirPath)
        if not os.path.isdir(absDirPath):
            if os.path.isfile(absDirPath):
                raise Exception("Input should be a dirPath, not a file!")
            else:
                raise Exception("Unknown error!")

        if not os.path.exists(dirPath):
            raise Exception("dirPath not exist!")

        names = os.listdir(absDirPath)
        curNames = []
        for name in names:
            curNames.append(name)
        return curNames

    @staticmethod
    def getAllFilePathsInPath(dirPath: str) -> list:
        """
        input: 
            dirPath -> str: directory you want to get files from
        return: 
            list : list of paths of all files in dirPath
        """
        absDirPath = os.path.abspath(dirPath)
        if not os.path.isdir(absDirPath):
            if os.path.isfile(absDirPath):
                raise Exception("Input should be a dirPath, not a file!")
            else:
                raise Exception("Unknown error!")

        if not os.path.exists(dirPath):
            raise Exception("dirPath not exist!")

        names = os.listdir(absDirPath)
        curNames = []
        for name in names:
            curNames.append(os.path.abspath(os.path.join(absDirPath, name)))
        return curNames


class CreateDirInPath:
    def __init__(self, dirPath=""):
        super(CreateDirInPath, self).__init__()
        self.dirPath = dirPath

    @staticmethod
    def createDirInPath(self, dirPath="") -> bool:
        if not dirPath:
            dirPath = self.dirPath

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
            print("Create directory success!")
            return True
        else:
            print("dirPath already exists!")
            return False
        return False

if __name__ == "__main__":
    pass
