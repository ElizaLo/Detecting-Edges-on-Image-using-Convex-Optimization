#include <QtWidgets>
#include <functional>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "algorithms.h"
#include "refedges.h"


using namespace std;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    scene = new QGraphicsScene(this);
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    filename = QFileDialog::getOpenFileName(this, tr("Select a single image"));
    original = QImage(filename);
    display(original);
    grayscale = original.convertToFormat(QImage::Format_Grayscale8);
    ui->comboBox->setCurrentIndex(0);
}


void MainWindow::display(const QImage& image) {
    scene->clear();
    scene->addPixmap(QPixmap::fromImage(image));
    scene->setSceneRect(image.rect());
    ui->image->setScene(scene);
    ui->image->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
    current = image;
}


void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    switch (index) {
    case 0:
        display(original);
        break;
    case 1:
        display(canny(grayscale, 1, 8, 13));
        break;
    case 6:
        display(refEdges (original, grayscale, 1, 8, 13, 1, 1, 5, 7, 0.5));
        break;
    default:
        function<QImage(const QImage&)> functions[] = {
            sobel,
            prewitt,
            roberts,
            scharr
        };
        display(functions[index - 2](grayscale));
        break;
    }
}


void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    ui->image->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
}


void MainWindow::on_pushButton_2_clicked()
{
    auto path = QFileDialog::getExistingDirectory(this, tr("Destination"));
    auto name = QString("%1_%2").arg(ui->comboBox->currentText(), QFileInfo(filename).fileName());
    current.save(QDir(path).filePath(name));
}
