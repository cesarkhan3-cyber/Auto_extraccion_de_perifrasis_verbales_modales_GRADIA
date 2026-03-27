# Auto_extraccion_de_perifrasis_verbales_modales_GRADIA
Este repositorio ofrece un script de Python que ayuda a llevar a cabo una extracción automática de 6 perífrasis verbales modales (PPVV) ("deber + INF", "deber de + INF", "tener de + INF", "tener que + INF", "haber que + INF" y "haber de + INF") desde archivos de textos planos (.txt). Las concortancias extraidas se vuelcan en una tabla Excel en la que se muestran los metadatos, las perífrasis, acompañadas del contexto lingúísticos anterior y posterior. El script también permite un análisis colostruccional (Collexeme Analysis) con el fin de mostrar la atracción/repulsión entre los verbos infinitivos y las PPVV. Este script todavía se está desarrollando, en el futuro permitirá una selección más amplia de otras perífrasis verbales. Está desarrollado por Yujian Han, mienbro del Equipo GRADIA.

# Requisitos previos
Para que funcione el script, hay que preparar una carpeta denominada "raw_txt_clean" en el mismo directorio en el que se coloca el script. En esa misma carpeta, hay que situar los archivos que se van a procesar. 

*En teoría también hay que preparar archivo de diccionario (en este script es "hsms.src"). Pero este diccionario no está cargado al repositorio. Así que si no lo tiene, durante el procesamiento aparecerá una alerta sobre la precisión. El "hsms.src" está basado de Freeling. Por eso, si es necesario, también puede compilar un propio y luego cambiar un poquito el contenido del script. 

# Instalar las dependencias de Python


pip install -r requirements.txt


python -m spacy download es_core_news_md

# Manejo y uso
Cualquier herramienta que apoya al entorno de Python. Corre el script:


python auto_extraccion v1.py


Se corre automáticamente todo el procesamiento y al final nos salen 3 archivos de .xslx, son (i) frecuencias de las PPVV con concordancias, (ii) los verbos infinitivos detectados y filtrados del corpus y (iii) resultados del análisis colostruccional.  

**Usamos spaCy y el archivo de diccionario para seleccionar y filtrar los verbos infinitivos que van a usar para el analisis colostruccional. En este sentido, se va a encontrar que en uno de los archivos .xslx de resultado "Baseline_strict_candidates_patched" hay varias hojas de trabajo. Donde se indica los verbos que no aproban esta doble seguridad. Si se observa que algunos verbos infinitivos que aparecen en este achivo son fiables, se puede añadirlos manualmente en el "supplement_inf_lemmas.txt", así la próxima vez cuando se corre el script, estos infinitivos aprobarán el filtro.

