import fileinput


class PhpHelper:
    def __init__(self):
        pass

    __config_path = 'kcaptcha/kcaptcha_config.php'

    def set_alpabet(self, alphabet):
        for line in fileinput.input(self.__config_path, inplace=True):
            match = '$allowed_symbols'
            if match in line and not '//' in line:
                line = "%s = \"%s\";\n" % (match, ''.join(alphabet))
            print(line, end="")

    def set_length(self, length):
        for line in fileinput.input(self.__config_path, inplace=True):
            match = '$length'
            if match in line and not '//' in line:
                if isinstance(length, int):
                    line = "%s = %d;\n" % (match, length)
                if isinstance(length, tuple):
                    line = "%s = mt_rand%s;\n" % (match, str(length))
            print(line, end="")
