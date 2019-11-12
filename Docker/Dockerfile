FROM ubuntu:latest

# Set environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install inital packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential pkg-config wget automake libtool && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install standard packages
RUN pip install --upgrade pip
RUN pip install glances[gpu]
RUN pip install jupyterlab
RUN pip install black
# Needed for Python refactoring:
RUN pip install rope
# Needed for Jupyter debugging:
RUN pip install ptvsd 
RUN pip install pylint

# Data Science packages
RUN pip install scikit-learn
RUN pip install xgboost
RUN pip install pandas
RUN pip install seaborn

# Setup Git
RUN git config --global user.email $EMAIL

# Setup ~/.bashrc
RUN echo 'export PS1="[\[$(tput sgr0)\]\[\033[38;5;166m\]\u\[$(tput sgr0)\]\[\033[38;5;15m\]][\[$(tput sgr0)\]\[\033[38;5;14m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]]\n\\$ \[$(tput sgr0)\]"' >>  ~/.bashrc

CMD [ "/bin/bash", "-D" ]