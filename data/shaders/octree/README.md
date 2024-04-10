Notas de optimizacion, ideas y locuras varias:

    - Raymarching: unificar el numero de pasos para ponerselo facil al scheduler.

Write_to_texture:
    Radeon graphic profiler: 
        VALU (Vector Arithmetic Logic Unit) uso 100%
        SALU (Scalar Arithmetic Logic Unit) uso 23.8%
        SMEM (Scalar Memory Operations) use 9.6%

    Lo usyo seria subsanar el uso del VALU
        Hay varias instrucciona dentro del primer branch del shader 'v_cvt_f32_u32_e32'
        que cuestan mas de 20,000 clocks de GPU, seguramente porque estan saturando el VALU
        Esta instruccion convierte un u32 en f32 y lo guarda en un resgitor de vector (forma de vector creo)


    Nsight
        SM Occupany esta sufriendo
        A partir de N numero de edits, se puede tener un write_to_texture de un brick entero
        Y un write_to_texture que solo haga medio brick, usando varios hilos para dividirse el trabajo