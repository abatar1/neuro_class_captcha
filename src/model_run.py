import model_generator as mg
import php_helper as ph

alphabet = "23456789abcdegkmnpqsuvxyz"
alphabet_size = len(alphabet)

generator = mg.ModelGenerator()
helper = ph.PhpHelper()
helper.set_alpabet(alphabet)
min_size, max_size = 1, 10

for i in range(min_size, max_size + 1):
    helper.set_length(i)
    generator.generate_text_rec(i, alphabet_size, 1000 * alphabet_size, open('./key', 'r').read(), str(i))

helper.set_length((min_size, max_size))
generator.generate_text_rec(i, alphabet_size, 1000 * (max_size - min_size + 1), len(open('./key', 'r').read()), 'num')