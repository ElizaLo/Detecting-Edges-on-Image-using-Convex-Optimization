#-------------------------------------------------
#
# Project created by QtCreator
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = edge-detection
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    algorithms.cpp \
    refedges.cpp

HEADERS  += mainwindow.h \
    kernels.h \
    algorithms.h \
    refedges.h \
    matrix.h

FORMS    += mainwindow.ui

CONFIG += c++14
QMAKE_CXXFLAGS += -Wall
