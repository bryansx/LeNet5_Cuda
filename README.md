# LeNet5_Cuda
TP Hardware for Signal Processing

## Partie 1
1- Creation de fonctions basiques de manipulations de matrices en C et en Cuda

2- Comparaison de temps de calcul pour l'addition parallélisé et non parallélisé avec "addMatC" et "addMatCuda" qui prennent en argument la taille des matrices.

    2.1- Pour des matrices de taille 10000*1000 on obtient une accélération de (45604/22), donc d'environ 2000. Ce qui reste dans l'ordre de grandeur de l'accélération th. 