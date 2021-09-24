
def getDfAndEncoder(df, dictCatVarAndReferences, flagUseStatsFormulaNameLike=False):
    """Categoriza un df usando como base un diccionario con columnas y referencias. Esta referencia es eliminada
    luego del encodeo.

    Parameters:
    df (dataframe): df con variables categoricas
    dictCatVarAndReferences (dict): diccionario en donde {key:value} = {nombreColumna:valorReferencia}

    Returns:
    df: df encodeado
    encoder:  OneHotEncoder instanciado
   """

    # validar que existan todas las columnas
    if not set(dictCatVarAndReferences.keys()).issubset(df.columns):
        colNoExists = set(dictCatVarAndReferences.keys()) - set(df.columns)
        raise Exception(f"Las columnas {colNoExists} no existen en el df")

    for k in dictCatVarAndReferences:
        # validar que existan las las referencias
        if not (df[k] == dictCatVarAndReferences[k]).any():
            raise Exception(f"El valor {dictCatVarAndReferences[k]} no existe en la columna {k}")

    #     valRefToDrop, categoricalCols, colNamesToEncode  = []
    #     for colName,valRef in dictCatVarAndReferences
    #         valRefToDrop.append(colName)
    #         categoricalCols.append(valRef)
    #         colNamesToEncode.append('C(' + colName + ', Treatment(' + "'" + valRef + "'" + '))')

    valRefToDrop = [value[1] for value in dictCatVarAndReferences.items()]
    categoricalCols = [value[0] for value in dictCatVarAndReferences.items()]
    colNamesToEncode = categoricalCols
    if flagUseStatsFormulaNameLike:
        colNamesToEncode = 'C(' + colName + ', Treatment(' + "'" + valRef + "'" + '))'

    cat_encoder = OneHotEncoder(sparse=False, drop=valRefToDrop)
    dfEncodedCatVars = pd.DataFrame(cat_encoder.fit_transform(df[categoricalCols]),
                                    columns=cat_encoder.get_feature_names(colNamesToEncode))

    return dfEncodedCatVars, cat_encoder


def getDfEncoded(encoder, df, dictCatVarAndReferences):
    """Encodea un df usando

    Parameters:
    encoder: OneHotEncoder instanciado
    df (dataframe): df con variables categoricas
    dictCatVarAndReferences (dict): diccionario en donde {key:value} = {nombreColumna:valorReferencia}

    Returns:
    df: df encodeado
    """
    # validar que existan todas las columnas
    if not set(dictCatVarAndReferences.keys()).issubset(df.columns):
        colNoExists = set(dictCatVarAndReferences.keys()) - set(df.columns)
        raise Exception(f"Las columnas {colNoExists} no existen en el df")

    for k in dictCatVarAndReferences:
        # validar que existan las las referencias
        if not (df[k] == dictCatVarAndReferences[k]).any():
            raise Exception(f"El valor {dictCatVarAndReferences[k]} no existe en la columna {k}")

    categoricalCols = [value[0] for value in dictCatVarAndReferences.items()]
    dfEncodedCatVars = pd.DataFrame(encoder.transform(df[categoricalCols]),
                                    columns=encoder.get_feature_names(categoricalCols))

    return dfEncodedCatVars
