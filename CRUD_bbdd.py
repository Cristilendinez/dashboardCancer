import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import settings


nameBBDD = "CancerMama"

############## Create database in postgress ###########

class crud:
    def createDatabase(nameBBDD):
        conn = psycopg2.connect(user=settings.USER,
                                password=settings.PASSWORD,
                                host=settings.HOST,
                                port=settings.PORT)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cur = conn.cursor()

        try:
            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(nameBBDD)))
        except psycopg2.Error as e:
            return str(e)

        cur.close()
        conn.close()
        return f"{nameBBDD} Database created successfully........"


    def createTabla(nameBBDD, tabla):       
        conn = psycopg2.connect(user=settings.USER,
                                password=settings.PASSWORD,
                                host=settings.HOST,
                                port=settings.PORT)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cur = conn.cursor()
        try: 
            query=f"CREATE TABLE {tabla}(ID SERIAL PRIMARY KEY, diagnosis varchar(80), radius_mean real, texture_mean real, perimeter_mean real, area_mean real,smoothness_mean real, compactness_mean real, concavity_mean real, concave_points_mean real, symmetry_mean real, fractal_dimension_mean real,radius_se real, texture_se real, perimeter_se real, area_se real, smoothness_se real, compactness_se real, concavity_se real, concave_points_se real, symmetry_se real, fractal_dimension_se real, radius_worst real, texture_worst real, perimeter_worst real, area_worst real, smoothness_worst real, compactness_worst real, concavity_worst real, concave_points_worst real, symmetry_worst real, fractal_dimension_worst real);"
            cur.execute(query)
        except psycopg2.Error as e:
            print("Error al crear la tabla: %s" %str(e))
        conn.commit()
        cur.close()
        conn.close()



    def insertarDatos(tabla, tuplax): 
        # n_table:str,id:int, PL:float, PW:float, SL:float, SW:float, tipo:str
        conn = psycopg2.connect(user=settings.USER,
                                password=settings.PASSWORD,
                                host=settings.HOST,
                                port=settings.PORT)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        cols = '(diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean,smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst)'
        
        try:
            query = f"INSERT INTO {tabla} {cols} VALUES{tuplax};" #(tabla, cols, tuplax)
            cur.execute(query) 
        except psycopg2.Error as e:
            print("Error al insertar dato: %s" %str(e))

        conn.commit()

        cur.close()
        conn.close()

    def mostrar(tabla, column, val1, val2):
        conn = psycopg2.connect(user=settings.USER,
                                password=settings.PASSWORD,
                                host=settings.HOST,
                                port=settings.PORT)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        try:
            query = f"SELECT * FROM {tabla} WHERE {column} BETWEEN {val1} AND {val2};"
            cur.execute(query)
            rows = cur.fetchall()

            for row in rows:
                print(row)             

        except psycopg2.Error as e:
            error = "Error mostrar registros: %s" + str(e)
            cur.close()
            conn.close()  
            return {f"msg":error}

        conn.commit()

        cur.close()
        conn.close()
        
        return rows

    # def actualizar(nameBBDD):
    #     cur, conn = connect()
    #     try:
    #         cur.execute("UPDATE notas1 SET grades=7.9 WHERE name='Juanito Perez';") 
    #     except psycopg2.Error as e:
    #         print("Error al actualizar dato: %s" %str(e))    

    #     conn.commit()

    #     cur.close()
    #     conn.close()

    def eliminar(tabla):
        conn = psycopg2.connect(user=settings.USER,
                                password=settings.PASSWORD,
                                host=settings.HOST,
                                port=settings.PORT)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        try:
            query = f"DELETE FROM {tabla};"
            cur.execute(query)
        except psycopg2.Error as e:
            print("Error al eliminar dato: %s" %str(e))    

        conn.commit()

        cur.close()
        conn.close()
        
    # # =============== Código, leer los datos mediante CMD: =============== 
    # PS C:\Users\Usuario\Documents\DATA_SCIENCE\FEI\Git_Repositorios\Docker_Postgres> docker exec -it docker_postgres-db-1 bash
    # root@e1e6287130bd:/# psql -U Maria actividad
    # actividad=# SELECT * FROM edición;
    # actividad=# SELECT * FROM notas;
    # actividad=# SELECT *  FROM notas WHERE notas BETWEEN 5 AND 6.5;
    # actividad=# SELECT *  FROM notas WHERE id_edic = 2;