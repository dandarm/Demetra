


def get_train_test_validation_df(tracks_df, percentage=0.7, validation_percentage=0.15, id_col='id_final', verbose=True):
    """
    Divide il DataFrame tracks_df in tre parti: train, test e validation.
    percentage è la percentuale di dati da usare per il training.
    validation_percentage è la percentuale di dati da usare per la validazione.
    """

    cicloni_unici = tracks_df[id_col].unique()    
    len_p = int(percentage*cicloni_unici.shape[0])
    cicloni_unici_train = cicloni_unici[:len_p]

    if validation_percentage > 0 :
        len_t = int((percentage + validation_percentage) * cicloni_unici.shape[0])
        cicloni_unici_test = cicloni_unici[len_p:len_t]
        cicloni_unici_validation = cicloni_unici[len_t:]
        print(f"Cicloni nel train: {cicloni_unici_train.shape[0]}, cicloni nel test: {cicloni_unici_test.shape[0]}, cicloni nella validation: {cicloni_unici_validation.shape[0]}")
    else: # per esempio == -1 e quindi non voglio un terzo set di validation
        cicloni_unici_test = cicloni_unici[len_p:]
        cicloni_unici_validation = []
        print(f"Cicloni nel train: {cicloni_unici_train.shape[0]}, cicloni nel test: {cicloni_unici_test.shape[0]}")

    tracks_df_train = tracks_df[tracks_df[id_col].isin(cicloni_unici_train)]    
    tracks_df_test = tracks_df[tracks_df[id_col].isin(cicloni_unici_test)]
    tracks_df_validation = tracks_df[tracks_df[id_col].isin(cicloni_unici_validation)]

    if verbose:
        print(f"Train rows: {tracks_df_train.shape[0]}, Test rows: {tracks_df_test.shape[0]}, Validation rows: {tracks_df_validation.shape[0]}")
        u_train = tracks_df_train.groupby(tracks_df_train[id_col]).apply('first')  
        print((u_train.end_time - u_train.start_time).sum())
        u_test = tracks_df_test.groupby(tracks_df_test[id_col]).apply('first')  
        print((u_test.end_time - u_test.start_time).sum())
        u_val = tracks_df_validation.groupby(tracks_df_validation[id_col]).apply('first')  
        print((u_val.end_time - u_val.start_time).sum())

    return tracks_df_train, tracks_df_test, tracks_df_validation