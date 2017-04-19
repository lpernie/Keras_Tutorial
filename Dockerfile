FROM andrewosh/binder-base

MAINTAINER Andrew Osheroff <andrewosh@gmail.com>

USER root
RUN pip install -U pip

USER main
RUN pip install -U deepdish
RUN pip install -U scikit-learn
RUN pip install -U keras
RUN pip install -U tensorflow
