# Auto_extraccion_de_perifrasis_verbales_modales_GRADIA
Este repositorio ofrece un script de python que ayuda a llevar a cabo una extracción automática de 6 perífrasis verbales modales ("deber + INF", "deber de + INF", "tener de + INF", "tener que + INF", "haber que + INF" y "haber de + INF") desde archivos de textos planos (.txt). Se calcula la frecuencia de las perífrasis verbales elegidas con concordancia de dichas PPVV para que se vuelva a revisar los resultados. También ofrece un análisis colostruccional (Collexeme Analysis) con el fin de mostrar la atracción/repulsión entre los verbos infinitivos y las PPVV.

# Requisitos previos
Para que funcione el script, hay que preparar dos carpetas: una que se llama "outputs" y la otra "raw_txt_clean" 


pip install -r requirements.txt


python -m spacy download es_core_news_md


El archivo hsms.src todavía no está cargado al repositorio. Es un diccionario basado de Freeling. Si es necesario también puede compilar un propio. 
