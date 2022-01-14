# LeNet5_Cuda
TP Hardware for Signal Processing

## Partie 1
1.1- Creation de fonctions basiques de manipulations de matrices en C et en Cuda

1.2- Comparaison de temps de calcul pour l'addition parallélisée et non parallélisée avec "addMatC" et "addMatCuda" qui prennent en argument la taille des matrices.

1.2.1- Pour des matrices de taille 1000*1000 on obtient une accélération de (10.6/0.1), donc d'environ 100. Ce qui est inférieur à l' accélération théorique de 1000. Cela est surement  a temps de copie des variables. 

1.3- Comparaison de temps de calcul pour la multiplication parallélisée et non parallélisée avec "mulMatC" et "mulMatCuda" qui prennent en argument la taille des matrices.

1.3.1 On obtient une accélération bien plus importante pour les multiplications de matrices: 8100 ms pour le CPU contre 0.8 pour le GPU, soit une accélération de 10000 environ.

## Partie 2

2.1- Nous allons dans un premier temps écrire et tester la fonction de convolution afin de s'assurer quelle se comporte comme prévu. Pour se faire nous utiliser une image de la database MNIST. Puis nous allons faire une convolution avec le filtre de Sobel afin de vérifier que nous avons une bonne détection des contours. (run ./test_conv2d_MNIST et ./test_conv2d_squarre).
On peut bien voir sur les resultats de la conv du carré que les lignes verticales en disparues et les contours horizontaux ont été détectés.

=> La fonction conv2d semble fonctionner comme il faut.

2.2- Travail sur le fichier "Partie2/CNN.cu" Layer 2: Convolution 2D avec 6 noyaux de conv de taille 5x5. La taille résultantes est donc de 6x28x28.

2.3 Layer 3: Sous-échantiollonnage par moyennage 2x2 => donne une matrice en sortie de taille 6x14x14.

2.4 Fonction d'activation tanh. L'activation se fait juste apres la convolution.

## Partie 2: Un peu de Python 

Dans cette partie nous allons créer un réseau LeNet5 sur tensorflow puis l'entrainer sur le dataset MNIST. Une fois le modèle entraîné nous allons exporter les poids obtenus sur notre code Cuda pour pouvoir réaliser des prédictions avec GPU.

3.1 Il nous manque la deuxiéme convolution (+ activation), l'averagePooling, le Flatten et le réseau Fully Connected de fin.
Les couches Flatten et Dense sont "facilements" parallélisables.
Il manque aussi la fonction d'activation softmax.

3.1.1 Flatten: Puisque nous avons travaillé depuis le début avec des vecteurs, nous n'avons pas besoin de fonction Flatten :)))

3.1.2 Dense: Les couches denses consistent tout simplement a faire des muliplication matricielles A*v, avec A la matrice des poids et v le vecteur d'entrée.

3.2 Une fois le réseau entier créé il faut importer les poids stockés dans le fichier "my.h5".





