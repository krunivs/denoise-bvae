import shutil
import json
import os


class FileUtil:
    @staticmethod
    def move_file(src, dst):
        """
        move file
        :param src: (str) source file path
        :param dst: (str) destination file path
        :return:
        """
        if not os.path.isfile(src):
            raise FileNotFoundError('Not found {} file'.format(src))

        shutil.move(src, dst)

    @staticmethod
    def create_directory(abs_dir_path):
        """
        desc: create directory
        arg0: (string) absolute directory path
        exception: OSError, ValueError
        """
        if abs_dir_path is None:
            raise ValueError('Error: directory path is None')
        try:
            if os.path.exists(abs_dir_path) is False:
                os.makedirs(abs_dir_path)
        except OSError:
            raise OSError('Error: Creating directory. ' + abs_dir_path)

    @staticmethod
    def delete_directory(dir_path):
        """
        delete directory
        :param dir_path: (string) deleting directory
        :return:
        """

        shutil.rmtree(dir_path)

    @staticmethod
    def save_file_content_to_bin_file(file_name, file_content):
        """
        desc: save web upload file as binary file
        arg0: (string) filename;
        arg1: (string) file_content; ex) file_content = ContentFile(http_file_object.read())
        exception: OSError, ValueError
        """
        f = open(file_name, 'wb')
        for chunk in file_content.chunks():
            f.write(chunk)
        f.close()

    @staticmethod
    def delete_file(file_path):
        """
        delete file
        :param file_path: (str) deleting file path
        :return:
        """
        if os.path.isfile(file_path):
            os.remove(file_path)

    @staticmethod
    def to_json_file(content, filename):
        """
        save json data to file
        :param content: (dict) content
        :param filename: (string) filename
        :return:
        """
        json_dumps_str = json.dumps(content, indent=4)
        f = open(filename, 'w')
        f.write(json_dumps_str)

        f.close()

    @staticmethod
    def from_json_file(filename):
        """
        load json data from file
        :param filename: (string)
        :return:
        """
        f = open(filename, 'r')
        json_dumps_str = json.load(f)
        f.close()

        return json_dumps_str

    @staticmethod
    def read_text_file(filename):
        """
        read
        :param filename: (str) file path
        :return:
        """
        text_file = open(filename, 'r')
        data = text_file.read()
        text_file.close()

        return data

    @staticmethod
    def write_text_file(filename, content):
        """
        write file
        :param filename: (str) file name
        :param content: (str) file content
        :return:
        """
        text_file = open(filename, 'w')
        text_file.write(content)
        text_file.close()
