import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')

import cv2 as cv


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

import Basics
import Texture_Computation as texture




def testing(k):
    """

    :param k:
    :return:
    """

    model_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Model/"
    columnas_caracteristicas = ["Energia", "Contraste", "Homogeneidad", "Entropia", "Disimilaridad"]

    acc_values = []
    fscore_values = []
    for fold in range(k):
        print("Fold : ", fold)
        # Reading good instances
        y_real = [] # To append lables of the fold
        all_bad = []
        all_good = []

        for ki in range(k):
            if ki == fold:
                # Reading good instances
                split_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/"
                with open(split_path + "Bad_K" + str(ki) + ".txt", 'r', encoding='utf-8') as file:
                    list_bad = [linea.strip() for linea in file]

                split_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/"
                with open(split_path + "Good_K" + str(ki) + ".txt", 'r', encoding='utf-8') as file:
                    list_good = [linea.strip() for linea in file]

                for j in list_bad:
                    all_bad.append(j)

                for j in list_good:
                    all_good.append(j)

                # Labels assigning
                for i in range(len(all_bad)):
                    y_real.append(0)
                for i in range(len(all_good)):
                    y_real.append(1)

                # Reading DataFrame
                feat_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Features/bads/features.csv"
                df = pd.read_csv(feat_path, index_col=0)
                # print(df.head())

                # Filtrar el DataFrame
                df_filtrado_bad = df[df["Image"].isin(all_bad)]
                # print(df_filtrado_bad)
                path2save = model_path + "TestingBad_K" + str(fold) + ".csv"
                df_filtrado_bad.to_csv(path2save, index_label=0)
                X_bad = df_filtrado_bad[columnas_caracteristicas].values

                feat_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Features/goods/features.csv"
                df = pd.read_csv(feat_path, index_col=0)
                df_filtrado_good = df[df["Image"].isin(all_good)]
                # print(df_filtrado_good)
                path2save = model_path + "TestingGood_K" + str(fold) + ".csv"
                df_filtrado_good.to_csv(path2save, index_label=0)
                X_good = df_filtrado_good[columnas_caracteristicas].values

                X_test = np.concatenate((X_bad, X_good), axis=0)

                # Open model to predict
                # Reading model of K
                with open("C:/Users/alber/Documents/Textile_defects/Project_Data/Model/Model_K" + str(fold) + ".sav", "rb") as f:
                    model = pickle.load(f)

                y_predicted = None
                if model is not None:
                    y_predicted = model.predict(X_test)
                #print(y_predicted)

                accuracy = accuracy_score(y_real, y_predicted)
                print(f"Accuracy: {accuracy:.3f}")  # Prints with 3 decimal places

                fscore = f1_score(y_real, y_predicted)
                print(f"F1 score: {fscore:.3f}")  # Prints with 3 decimal places



                # Evaluation metrics by fold computed

                acc_values.append(accuracy)
                fscore_values.append(fscore)


    print(f"Accuracy average: {np.array(acc_values).mean():.3f}")
    print(f"F1 score average: {np.array(fscore_values).mean():.3f}")




    # Plotting the results

    x = np.array(["Accuracy", "F1-score"])
    y = np.array([np.array(acc_values).mean(), np.array(fscore_values).mean()])

    plt.bar(x, y, color="red", width = 0.25)
    plt.title('Resultados de Accuracy y F1-score')
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

    return None



def training(k):
    """
    Creates a model using SVM technique. One model per fold is calculated
    :return:
    """
    model_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Model/"
    Basics.lib_create_folder(model_path)

    columnas_caracteristicas = ["Energia", "Contraste", "Homogeneidad", "Entropia", "Disimilaridad"]

    for fold in range(k):
        print("Fold : ", fold)
        # Reading good instances
        y = [] # To append lables of the fold
        all_bad = []
        all_good = []

        for ki in range(k):
            if ki != fold:
                # Reading good instances
                split_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/"
                with open(split_path + "Bad_K" + str(ki) + ".txt", 'r', encoding='utf-8') as file:
                    list_bad = [linea.strip() for linea in file]

                split_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/"
                with open(split_path + "Good_K" + str(ki) + ".txt", 'r', encoding='utf-8') as file:
                    list_good = [linea.strip() for linea in file]

                #print(list_bad)

                for j in list_bad:
                    all_bad.append(j)

                for j in list_good:
                    all_good.append(j)

        #print(len(all_good))
        #print(len(all_bad))

        # Labels assigning
        for i in range(len(all_bad)):
            y.append(0)
        for i in range(len(all_good)):
            y.append(1)

        # Reading DataFrame
        feat_path ="C:/Users/alber/Documents/Textile_defects/Project_Data/Features/bads/features.csv"
        df = pd.read_csv(feat_path, index_col=0)
        #print(df.head())

        # Filtrar el DataFrame
        df_filtrado_bad = df[df["Image"].isin(all_bad)]
        #print(df_filtrado_bad)
        path2save = model_path + "TrainingBad_K" + str(fold) + ".csv"
        df_filtrado_bad.to_csv(path2save, index_label=0)
        X_bad = df_filtrado_bad[columnas_caracteristicas].values

        feat_path = "C:/Users/alber/Documents/Textile_defects/Project_Data/Features/goods/features.csv"
        df = pd.read_csv(feat_path, index_col=0)
        df_filtrado_good = df[df["Image"].isin(all_good)]
        #print(df_filtrado_good)
        path2save = model_path + "TrainingGood_K" + str(fold) + ".csv"
        df_filtrado_good.to_csv(path2save, index_label=0)
        X_good = df_filtrado_good[columnas_caracteristicas].values

        X = np.concatenate((X_bad, X_good), axis=0)
        #print(X)


        svm_model = SVC(kernel='rbf',  C=1.0, gamma='scale')
        svm_model.fit(X, y)
        print("Model constructed...")


        with open(model_path + "/Model_K" + str(fold) + ".sav", "wb") as f:
            pickle.dump(svm_model, f)

    return None



def feature_extraction():
    """
    Function to compute Co-ocurrence matrix. Then we compute the feature vector from it.
    :return:
    """
    path = "C:/Users/alber/Documents/Textile_defects/Project_Data/"

    # Creating folder
    Basics.lib_create_folder(path + "Features")

    # Creating features good/bad folder
    Basics.lib_create_folder(path + "Features/goods")
    Basics.lib_create_folder(path + "Features/bads")

    # Goods
    path_img = "C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/Binary_Classif/goods/"
    list_img = Basics.lib_read_from_folder(path_img)

    df = pd.DataFrame(columns=["Image", "Energia", "Contraste", "Homogeneidad", "Entropia", "Disimilaridad"])
    l_image = []
    l_energia = []
    l_contraste = []
    l_homog = []
    l_entropia = []
    l_disimil = []
    for i in list_img:
        print("Processing : ", i)
        img = Basics.img_read(path_img + i, cv.IMREAD_GRAYSCALE) # Reading image

        mtx = texture.co_ocurrence_matrix0(img, distance=5) # Computing Co-ocurrence

        energia, contraste, homog, entropia, disimil = texture.co_ocurrence_featureVector(mtx)

        l_image.append(i)
        l_energia.append(energia)
        l_contraste.append(contraste)
        l_homog.append(homog)
        l_entropia.append(entropia)
        l_disimil.append(disimil)

    df["Image"] = np.array(l_image)
    df["Energia"] = np.array(l_energia)
    df["Contraste"] = np.array(l_contraste)
    df["Homogeneidad"] = np.array(l_homog)
    df["Entropia"] = np.array(l_entropia)
    df["Disimilaridad"] = np.array(l_disimil)

    # Saving the dataframe
    df.to_csv(path + "Features/goods/features.csv")


    # Bads
    path_img = "C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/Binary_Classif/bads/"
    list_img = Basics.lib_read_from_folder(path_img)

    df = pd.DataFrame(columns=["Image", "Energia", "Contraste", "Homogeneidad", "Entropia", "Disimilaridad"])
    l_image = []
    l_energia = []
    l_contraste = []
    l_homog = []
    l_entropia = []
    l_disimil = []
    for i in list_img:
        print("Processing : ", i)
        img = Basics.img_read(path_img + i, cv.IMREAD_GRAYSCALE)  # Reading image

        mtx0 = texture.co_ocurrence_matrix0(img, distance=1)  # Computing Co-ocurrence
        mtx45 = texture.co_ocurrence_matrix45(img, distance=1)  # Computing Co-ocurrence
        mtx90 = texture.co_ocurrence_matrix90(img, distance=1)  # Computing Co-ocurrence
        mtx135 = texture.co_ocurrence_matrix135(img, distance=1)  # Computing Co-ocurrence

        mtx = (mtx0 + mtx45 + mtx90 + mtx135) / 4

        energia, contraste, homog, entropia, disimil = texture.co_ocurrence_featureVector(mtx)

        l_image.append(i)
        l_energia.append(energia)
        l_contraste.append(contraste)
        l_homog.append(homog)
        l_entropia.append(entropia)
        l_disimil.append(disimil)

    df["Image"] = np.array(l_image)
    df["Energia"] = np.array(l_energia)
    df["Contraste"] = np.array(l_contraste)
    df["Homogeneidad"] = np.array(l_homog)
    df["Entropia"] = np.array(l_entropia)
    df["Disimilaridad"] = np.array(l_disimil)

    # Saving the dataframe
    df.to_csv(path + "Features/bads/features.csv")

    return None



def randmly_selecting_Binary_clasification(n_goods, n_bad):
    """
    Selecting images of the two classes.
    :return:
    """

    bad_labels = ['hole', 'objects', 'oil spot', 'thread error']
    img_path = 'C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/'

    # bad ones
    list_hole = Basics.lib_read_from_folder(img_path + bad_labels[0])
    sample_hole = random.sample(list_hole, n_bad)
    list_objects = Basics.lib_read_from_folder(img_path + bad_labels[1])
    sample_objects = random.sample(list_objects, n_bad)
    list_oil = Basics.lib_read_from_folder(img_path + bad_labels[2])
    sample_oil = random.sample(list_oil, n_bad)
    list_thread = Basics.lib_read_from_folder(img_path + bad_labels[3])
    sample_thread = random.sample(list_thread, n_bad)

    # Creating folder to binary classification
    Basics.lib_create_folder(img_path + "Binary_Classif")
    Basics.lib_create_folder(img_path + "Binary_Classif/goods")
    Basics.lib_create_folder(img_path + "Binary_Classif/bads")

    # Orgninazing the bad class
    for i in range(len(sample_hole)):
        # Reading image
        img = Basics.img_read(img_path + bad_labels[0] + "/" + sample_hole[i])
        # Writing image
        Basics.img_write(img_path + "Binary_Classif/bads/" + sample_hole[i], img)

        # Reading image
        img = Basics.img_read(img_path + bad_labels[1] + "/" + sample_objects[i])
        # Writing image
        Basics.img_write(img_path + "Binary_Classif/bads/" + sample_objects[i], img)

        # Reading image
        img = Basics.img_read(img_path + bad_labels[2] + "/" + sample_oil[i])
        # Writing image
        Basics.img_write(img_path + "Binary_Classif/bads/" + sample_oil[i], img)

        # Reading image
        img = Basics.img_read(img_path + bad_labels[3] + "/" + sample_thread[i])
        # Writing image
        Basics.img_write(img_path + "Binary_Classif/bads/" + sample_thread[i], img)

    # Organizing the good class
    # good ones
    list_hole = Basics.lib_read_from_folder(img_path + "good")
    sample_good = random.sample(list_hole, n_goods)

    for i in sample_good:
        # Reading image
        img = Basics.img_read(img_path + "good/" + i)
        # Writing image
        Basics.img_write(img_path + "Binary_Classif/goods/" + i, img)

    return None


def split_kFold(k=10):
    """
    We splot the data into n-folds to validate the proposal
    :param k:
    :return:
    """

    Basics.lib_create_folder("C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/")

    list_good = Basics.lib_read_from_folder(
        "C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/Binary_Classif/goods")
    list_bad = Basics.lib_read_from_folder(
        "C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/Binary_Classif/bads")

    n_items = int(len(list_good) / k)
    for n in range(k):
        # Organizing data
        k_seleccionados = random.sample(list_good, n_items)
        # Eliminar los elementos seleccionados de la lista original
        list_good = [x for x in list_good if x not in k_seleccionados]



        with open("C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/Good_K"+
                  str(n) + ".txt", "w") as archivo:
            archivo.writelines(elemento + "\n" for elemento in k_seleccionados)


    for n in range(k):
        # Organizing data
        k_seleccionados = random.sample(list_bad, n_items)
        # Eliminar los elementos seleccionados de la lista original
        list_bad = [x for x in list_bad if x not in k_seleccionados]



        with open("C:/Users/alber/Documents/Textile_defects/Project_Data/Validation/Bad_K"+
                  str(n) + ".txt", "w") as archivo:
            archivo.writelines(elemento + "\n" for elemento in k_seleccionados)

    return None