import json
import subprocess
from django.http import JsonResponse, HttpResponse
import optimalist.keras_model as keras_model
from .models import *
from django.views.decorators.csrf import csrf_exempt


def train_model(request=None):
    # TODO support for different types of intervals
    load_credentials()
    products = list(Product.objects.filter(interval=0).values_list('day').order_by('day').distinct())
    with open('keras-model/data/train-products-interval-0.txt', 'w') as file:
        for i in range(0, len(products), 4):
            temp = products[i:i+4]
            if len(temp) == 4:
                file_input = [str(x[0]) for x in temp]
                file.write(','.join(file_input))
                file.write('\n')

    # TODO increase sleep time
    # for linux
    subprocess.Popen("sleep 20 && python optimalist/keras_model.py --interval 0", shell=True)

    # for windows, but doesn't return immediately
    # subprocess.call("waitfor SomethingThatIsNeverHappening /t 20 2>NUL", shell=True)
    # subprocess.Popen("python optimalist/keras_model.py --interval 0", shell=True)

    if request:
        return HttpResponse(200)


@csrf_exempt
def get_prediction(request):
    # TODO support for different types of intervals
    if request.method == 'POST':
        load_credentials()
        data = json.loads(request.body)
        print("Data: {0}".format(data))
        x_input_dict = {}

        for title, day_list in data.items():
            day_list.sort(reverse=True)
            print("Title: {0}, Days: {1}".format(title, str(day_list)))
            products = list(Product.objects.filter(title=title).values_list('day').order_by('-day'))
            x_input = []
            for day in day_list:
                Product.objects.create(title=title, day=day, interval=0)
                if len(x_input) < 3:
                    x_input.append(day)
            input_size = len(x_input)
            if input_size < 3 <= input_size + len(products):
                for product in products[:3-input_size]:
                    x_input.append(product[0])
            x_input.reverse()
            print("X input: {0}".format(str(x_input)))
            if len(x_input) == 3:
                x_input_dict[title] = x_input

        response = {}
        if len(x_input_dict) > 0:
            keras_model.download_model()
            model = keras_model.load_model()
            for title, x_input in x_input_dict.items():
                prediction = keras_model.predict(model, x_input)[0][0]
                print("Prediction: {0}".format(prediction))
                response[title] = int(round(prediction))
        train_model()
        return JsonResponse(response, safe=False)

    return HttpResponse(404)


def evaluate_model(request):
    keras_model.download_model()
    model = keras_model.load_model()
    x_test, y_test = keras_model.get_data('keras-model/data/test-interval-0.txt', False)
    c = keras_model.evaluate(x_test, y_test, model)
    return JsonResponse(json.dumps(c.tolist()), safe=False)


def load_credentials():
    s = StorageCredentials.objects.first()
    data = {
        "type": s.type,
        "project_id": s.project_id,
        "private_key_id": s.private_key_id,
        "private_key": s.private_key.replace('\\n', '\n'),
        "client_email": s.client_email,
        "client_id": s.client_id,
        "auth_uri": s.auth_uri,
        "token_uri": s.token_uri,
        "auth_provider_x509_cert_url": s.auth_provider_x509_cert_url,
        "client_x509_cert_url": s.client_x509_cert_url
    }

    with open('./keras-model/credentials.json', 'w') as file:
        json.dump(data, file, indent=2)