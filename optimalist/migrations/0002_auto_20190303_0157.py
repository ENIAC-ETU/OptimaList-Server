# Generated by Django 2.0 on 2019-03-02 22:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('optimalist', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='interval',
            new_name='day',
        ),
        migrations.RenameField(
            model_name='product',
            old_name='date',
            new_name='modified_date',
        ),
        migrations.AddField(
            model_name='product',
            name='title',
            field=models.CharField(default='test', max_length=200),
            preserve_default=False,
        ),
    ]