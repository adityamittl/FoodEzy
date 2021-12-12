from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class orders(models.Model):
    order_id = models.AutoField(primary_key=True)
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    order_date = models.DateTimeField(auto_now_add=True)
    order_image = models.ImageField(upload_to='order_images', blank=True)

    def __str__(self):
        return str(self.order_id)
