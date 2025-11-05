## venv_cheat_sheet — Guide rapide pour environnements virtuels et notebooks

Objectif : créer un environnement virtuel à la racine d'un projet Python, y installer les dépendances, et connecter un notebook Jupyter / VS Code à cet environnement.

1) Créer et activer un virtualenv (Windows PowerShell)

```powershell
# Depuis la racine du projet
python -m venv .venv

# Activer le venv dans PowerShell
& .\.venv\Scripts\Activate.ps1

# Vérifier que vous utilisez le bon Python
python -c "import sys; print(sys.executable)"
```

Notes :
- `.venv` est un nom courant, pratique pour que VS Code le détecte automatiquement. Vous pouvez aussi choisir `env` ou `venv_project`.
- Ajoutez `.venv/` à votre `.gitignore` pour éviter d'ajouter l'environnement au dépôt.

2) Installer les dépendances

```powershell
# Installer des paquets individuellement
python -m pip install -U pip
python -m pip install -r requirements.txt

# Exemple minimal pour ce projet (si pas de requirements.txt)
python -m pip install langchain langchain-core langchain-openai python-dotenv
```

3) Préparer le kernel Jupyter (recommandé pour notebooks)

```powershell
# Installer ipykernel dans le venv
python -m pip install ipykernel

# Créer un kernel nommé (visible ensuite depuis Jupyter/VS Code)
python -m ipykernel install --user --name project-venv --display-name "Python (project-venv)"
```

Après cela, dans VS Code ouvrez votre notebook (`.ipynb`) puis, en haut à droite, sélectionnez le kernel "Python (project-venv)" ou sélectionnez l'interpréteur qui pointe vers `.venv\Scripts\python.exe`.

4) Astuce : installer depuis une cellule de notebook

Si vous préférez installer des paquets directement depuis le notebook (installe pour le kernel actif) :

```python
# dans une cellule du notebook
%pip install nom_du_paquet
```

5) Variables d'environnement (ex. OPENAI_API_KEY)

Option A — définir pour la session PowerShell (temporaire, utile pour tests) :

```powershell
$env:OPENAI_API_KEY = (Get-Content ..\OPENAI_API_KEY.txt).Trim()
# Puis démarrer VS Code ou le serveur Jupyter depuis cette session si besoin
```

Option B — charger depuis un fichier dans le code (déjà présent dans le projet) :

```python
OPENAI_API_KEY = open("../OPENAI_API_KEY.txt", "r").read().strip()
```

6) Vérifications rapides

```powershell
# vérifier les paquets installés
python -m pip list | Select-String "langchain|langchain-core|ipykernel"

# tester un import
python -c "import langchain, langchain_core; print(langchain.__version__, langchain_core.__version__)"
```

7) Règles pratiques / bonnes pratiques
- Utilisez un venv par projet pour éviter les conflits de dépendances.
- Mettez `.venv/` dans `.gitignore`.
- Préférez `python -m pip` plutôt que `pip` seul pour garantir l'installation dans le bon interpréteur.
- Créez un kernel ipykernel dédié pour chaque venv afin d'éviter d'avoir à deviner quel interpreter est utilisé par le notebook.

8) En cas de problème (diagnostic rapide)
- Lancer `python -c "import sys; print(sys.executable)"` pour confirmer le Python actif.
- Dans VS Code, vérifier le sélecteur d'interpréteur (coin supérieur droit du notebook) et redémarrer le kernel.
- Réinstaller `ipykernel` dans le venv si le kernel n'apparaît pas.

---

Si vous voulez, je peux aussi :
- ajouter automatiquement `.venv/` dans le `.gitignore` du projet (si vous voulez que je le fasse),
- créer un petit script `setup_env.ps1` qui exécute ces étapes automatiquement.
