import model_generator as mg
import php_helper as ph


sample_size = 5000
n_alphabet = [n for n in range(1, sample_size + 1)]
c_alphabet = list("23456789abcdegkmnpqsuvxyz")
min_size, max_size = 1, 6

generator = mg.ModelGenerator()
helper = ph.PhpHelper()
helper.set_alpabet(c_alphabet)


def save_model(model, model_result, model_name):
    model.save('models/' + model_name + '.h5')
    model.save_weights('weights/' + model_name + '_weights.h5')

    with open("model_result.txt", "a") as res_file:
        res_file.write(model_name + ': ' + ' '.join(str(e) for e in model_result) + '\n')


generator_path = 'kcaptcha/index.php'
for i in range(min_size, max_size + 1):
    helper.set_length(i)
    (model, model_result) = generator.generate(generator_path=generator_path,
                                               alphabet=c_alphabet,
                                               key_mode='str',
                                               sample_size=sample_size)
    save_model(model, model_result, model_name=str(i))

helper.set_length((min_size, max_size))
(model, model_result) = generator.generate(generator_path=generator_path,
                                           alphabet=n_alphabet,
                                           key_mode='len',
                                           sample_size=sample_size)
save_model(model, model_result, model_name='recognize_size')
