import streamlit as st

from render import render_bc, render_lr, render_mc

tasks = {
    'lr': 'Linear regression',
    'bc': 'Binary classification',
    'mc': 'Multi class classification'
}

task = st.selectbox(
    label='Select task',
    options=[
        tasks['lr'],
        tasks['bc'],
        tasks['mc']
    ]
)

if task == tasks['lr']:
    render_lr()
elif task == tasks['bc']:
    render_bc()
else:
    render_mc()
