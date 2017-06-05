import fileinput

class PhpHelper:
    __config_path = 'kcaptcha/kcaptcha_config.php'

    def set_alpabet(self, alphabet):
        for line in fileinput.input(self.__config_path, inplace=True):
            match = '$allowed_symbols'
            if match in line:
                line = "%s = \"%s\";\n" % (match, alphabet)
            print line,

    def set_length(self, length):
        for line in fileinput.input(self.__config_path, inplace=True):
            match = '$length'
            if match in line:
                if isinstance(length, int):
                    line = "%s = %d;\n" % (match, length)
                if isinstance(length, tuple):
                    line = "%s = mt_rand%s;\n" % (match, str(length))
            print line,