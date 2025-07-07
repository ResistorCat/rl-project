# rl-project
Project for the Reinforcement Learning course (2025-1)


```bash
py -m baseline.main
```


## Mejoras

### Contra quien se enfrenta

### Hiperparametros finetunnig

### Agrandar embedding

### Actualizar reward function

### Arreglar action_to_order y order_to_action
Para que detecte cuando puede o no hacer una accion. (Solo switch). Evitar `DefaultOrder`


### Que se entere de los pokemon vivos, muertos: que propague bien eso

### Asegurarme que siempre se mapee correctamente action (0 < action < 6) al pokemon correcto
Cuando hay pokemon debilitados. por ejemplo 1 debilitado se ve algo asi
-------------------
|--|--|--|--|--|--|
| Activo | Debilitado | Inactivo | Inactivo | Inactivo| Inactivo |
| 0 | 1 |2 |3|4|5|
| No Switch ya que activo | No Switch ya que muerto | Sí Switch | Sí Switch| Sí Switch | Sí Switch|
| no forma parte de `battle.available_switches`| no forma parte de `battle.available_switches` | si | si | si| si|


# Revival Blessing falla -> pide elegir un pokemon fallecido

# No maneja Encore

# No maneja disabled

# No maneja recharge