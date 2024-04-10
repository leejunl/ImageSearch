"""
Django settings for KerasImage project.

Generated by 'django-admin startproject' using Django 3.2.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-^i@c0%h*4ifv$8%g6g@f!_!rqfgh*it^g9(4pc6)%d#!@5-vqm'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['127.0.0.1','192.168.62.136','7f592c04.r8.cpolar.top']

SPYDER_URL = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1712751694833_R&pv=&ic=0&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=&ie=utf-8&sid=&word=%E7%86%8A%E7%8C%AB'
SPYDER_COOKIES = 'PSTM=1704779916; BAIDUID=AFAC04B3FC90B3369C7DFC374388D779:FG=1; BIDUPSID=E3B3070B37370961836FFCD124A813FD; BDUSS_BFESS=jZRZ3JPVVVLQlZ5QXZkdlkxTzIzeS1yR2NhSUFQcHlJOWUxUS1lSHBCNi1NUVZtSUFBQUFBJCQAAAAAAAAAAAEAAABalPyRzMDUstHMu9IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAL6k3WW-pN1lT; ZFY=n:Bbgp2X6jjhoELqQNAnojjpWYy7KYZwye1CCr9JRenA:C; BAIDUID_BFESS=AFAC04B3FC90B3369C7DFC374388D779:FG=1; __bid_n=18de9cacf104032afd6e35; H_PS_PSSID=40212_40080_40364_40352_40303_40376_40415_40310_40317_40487_40512; H_WISE_SIDS=40212_40080_40364_40352_40303_40376_40415_40310_40317_40487_40512; H_WISE_SIDS_BFESS=40212_40080_40364_40352_40303_40376_40415_40310_40317_40487_40512; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; RT="z=1&dm=baidu.com&si=wqv5e3xo5q&ss=lurojnpb&sl=0&tt=0&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=40y&ul=b5ygy&hd=b5yh6"; BCLID=10853470590192550069; BCLID_BFESS=10853470590192550069; BDSFRCVID=c7_OJeC62iLSWOrtQytkT9g6JlX_3v3TH6_ndwDdpL1S7CANgsEJEG0PgM8g0KAbufEqogKKXgOTHw0F_2uxOjjg8UtVJeC6EG0Ptf8g0f5; BDSFRCVID_BFESS=c7_OJeC62iLSWOrtQytkT9g6JlX_3v3TH6_ndwDdpL1S7CANgsEJEG0PgM8g0KAbufEqogKKXgOTHw0F_2uxOjjg8UtVJeC6EG0Ptf8g0f5; H_BDCLCKID_SF=tbCHoCLKtCL3fP36q4O_KICShUFs55bdB2Q-5KL-2lR6Sp745hQB5-P7eftqBt7vXb7qoMbdJf7_e-3ebb3dDqJWLPQwKbJ4WeTxoUJjBCnJhhvq-RotQJ_ebPRiXPb9QgbfopQ7tt5W8ncFbT7l5hKpbt-q0x-jLTnhVn0M5DK0HPonHjL2ej3y3j; H_BDCLCKID_SF_BFESS=tbCHoCLKtCL3fP36q4O_KICShUFs55bdB2Q-5KL-2lR6Sp745hQB5-P7eftqBt7vXb7qoMbdJf7_e-3ebb3dDqJWLPQwKbJ4WeTxoUJjBCnJhhvq-RotQJ_ebPRiXPb9QgbfopQ7tt5W8ncFbT7l5hKpbt-q0x-jLTnhVn0M5DK0HPonHjL2ej3y3j; MCITY=-42%3A75%3A; userFrom=null; indexPageSugList=%5B%22%E7%86%8A%E7%8C%AB%22%2C%22%E7%8B%97%22%2C%22%E5%8A%A8%E7%89%A9%22%5D; cleanHistoryStatus=0; ab_sr=1.0.1_Y2IzNWI3YWE2YjU0OWY3ODMyMGYyOGYwYzkzNWE1N2MzZDFiM2FjYzgxMzUwY2QxYmE4M2Y1ODM5NDI2N2I3OGQwYTkxNjEzYzU5MGIyZjM2ZjkzM2I2NDc5N2FlMTI5NDM3NjkxYWRkMjIzMjQ1ZWJlMGI5NDZjYjE5YjE3MmM0OGM0M2EwZWFjOGE3NDk4MDUyY2UxMTBhOTMxMzAwNg=='

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'chy'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    #'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'KerasImage.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'KerasImage.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/

STATIC_URL = '/static/'

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
