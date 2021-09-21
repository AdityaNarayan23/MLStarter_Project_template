FROM python:3.8.3-slim-buster
USER root
RUN apt-get update
RUN apt-get -y install gcc
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install happybase
COPY --chown=root:root ./process_HBase_login_data.py process_HBase_login_data.py
COPY --chown=root:root ./run_Kmeans.sh run_Kmeans.sh
RUN chmod +x process_HBase_login_data.py
CMD ["sh", "-c", "./run_Kmeans.sh"]
