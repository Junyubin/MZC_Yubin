# aws 에서 제공하는 lambda base image (python)
FROM amazon/aws-lambda-python:3.9

# optional : ensure that pip is up to data
RUN /var/lang/bin/python3.9 -m pip install --upgrade pip

# install packages
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install joblib==1.2.0

COPY lambda_function.py /var/task

# lambda_function.handler 실행
CMD ["lambda_function.handler"]
