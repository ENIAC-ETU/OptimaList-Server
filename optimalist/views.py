import sys
import re
import json

from django.shortcuts import render, redirect
from django.http import JsonResponse

from PIL import Image
import pyocr
import pyocr.builders


def get_ocr(request):
    tool = pyocr.get_available_tools()[0]

    txt = tool.image_to_string(
        Image.open('optimalist/static/fis.jpeg'),
        lang='eng',
        builder=pyocr.builders.TextBuilder()
    )

    products = re.findall('.+\s+\*.+', txt)

    return JsonResponse(json.dumps(products), safe=False)
