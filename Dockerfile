# Danny Clemens
#
# Dockerfile

# In order to make this program practical, I implemented MLOPS

FROM python:3.13

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install pandas scikit-learn matplotlib xgboost

# Run training script 
CMD ["python", "train.py"]