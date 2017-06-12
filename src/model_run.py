import model_generator as mg
import php_helper as ph

alphabet = "23456789abcdegkmnpqsuvxyz"
sample_size = 100
min_size, max_size = 1, 10

generator = mg.ModelGenerator()
helper = ph.PhpHelper()
helper.set_alpabet(alphabet)

def save_model(model, model_result, model_name):
    model.save('models/' + model_name + '.h5')
    model.save_weights('weights/' + model_name + '_weights.h5')

    with open("model_result.txt", "a") as file:
        file.write(model_name + ': ' + str(model_result))

for i in range(min_size, max_size + 1):
    helper.set_length(i)
    (model, model_result) = generator.generate(generator_path='kcaptcha/index.php',
                       alphabet=alphabet,
                       key_mode='str',
                       sample_size=sample_size)
    save_model(model, model_result, model_name=str(i))

helper.set_length((min_size, max_size))
(model, model_result) = generator.generate(generator_path='kcaptcha/index.php',
                                           alphabet=alphabet,
                                           key_mode='str',
                                           sample_size=sample_size)
save_model(model, model_result, model_name='num')