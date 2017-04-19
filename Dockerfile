FROM andrewosh/binder-base

MAINTAINER Andrew Osheroff <andrewosh@gmail.com>

USER main
# Upgrade pip
RUN pip install -U pip
# Keras
RUN pip install -U keras
RUN mkdir $HOME/.keras
RUN cp files/keras.json $HOME/.keras/
RUN pip install -U tensorflow
# Various
RUN pip install -U deepdish
RUN pip install -U scikit-learn
RUN pip install -U cmake
RUN pip install -U xrootd
RUN pip install -U fftw
RUN pip install -U openssl
RUN pip install -U gsl
# ROOT
RUN mkdir homebrew && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew
RUN brew install python
RUN brew install root6
