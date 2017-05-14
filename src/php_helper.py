class PhpHelper:
    @staticmethod
    def set_alpabet(self, alphabet):
        import fileinput

        for line in fileinput.input('kcaptcha/kcaptcha_config', inplace=True):
            match = '$allowed_symbols'
            if match in line:
                print "%s" % match + ' = "' + alphabet + '";',
                break

    @staticmethod
    def set_length(self, length):
        import fileinput

        for line in fileinput.input('kcaptcha/kcaptcha_config', inplace=True):
            match = '$length'
            if match in line:
                if isinstance(length, int):
                    print "%s" % match + ' = ' + length + ';',
                if isinstance(length, tuple):
                    print "%s" % match + ' = mt_rand' + length + ';',