from django.db import models
from django.contrib.auth.models import User


class profile(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)
    photo = models.ImageField(upload_to='profile_pictures')
    email = models.EmailField()

    def __str__(self):
        return str(self.user)

class sendotp(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)