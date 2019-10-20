# If .bash_profile exists, bash doesn't read .profile
if [[ -f ~/.profile ]]; then
  . ~/.profile
fi

# If the shell is interactive and .bashrc exists, get the aliases and functions
if [[ $- == *i* && -f ~/.bashrc ]]; then
    . ~/.bashrc
fi

# https://github.com/ipython/ipython/issues/10627
export QT_QPA_PLATFORM='offscreen'
