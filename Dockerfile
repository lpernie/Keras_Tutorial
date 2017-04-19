FROM andrewosh/binder-base

MAINTAINER Andrew Osheroff <andrewosh@gmail.com>

USER root

# Add dependencies
RUN pip install -U pip
RUN pip install -U scikit-learn
#RUN pip install -U sklearn
RUN pip install -U keras
#RUN pip install -U Tensorflow

USER main
