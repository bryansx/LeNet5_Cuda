# LeNet5_Cuda
TP Hardware for Signal Processing

## Partie 1
1- Creation de fonctions basiques de manipulations de matrices en C et en Cuda

2- Comparaison de temps de calcul pour l'addition parallélisée et non parallélisée avec "addMatC" et "addMatCuda" qui prennent en argument la taille des matrices.

2.1- Pour des matrices de taille 10000*1000 on obtient une accélération de (45604/22), donc d'environ 2000. Ce qui reste dans l'ordre de grandeur de l'accélération th. 

3- Comparaison de temps de calcul pour la multiplication parallélisée et non parallélisée avec "mulMatC" et "mulMatCuda" qui prennent en argument la taille des matrices.

3.1 On obtient une accélération bien plus importante pour les multiplications de matrices: 9512 ms pour le CPU contre 0.025 pour le GPU, soit une accélération de 38000 environ.

