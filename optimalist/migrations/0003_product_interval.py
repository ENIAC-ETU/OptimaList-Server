# Generated by Django 2.0 on 2019-03-03 12:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('optimalist', '0002_auto_20190303_0157'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='interval',
            field=models.IntegerField(default=0),
        ),
    ]
