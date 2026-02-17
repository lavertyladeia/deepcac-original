#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para testar o ambiente DeepCAC com Python 2.7
Verifica se todas as dependências estão instaladas corretamente
"""

import sys
import importlib

def test_import(module_name, version_attr=None, version_method=None):
    """Testa se um módulo pode ser importado e retorna sua versão"""
    try:
        module = importlib.import_module(module_name)
        if version_attr:
            version = getattr(module, version_attr)
        elif version_method:
            version = getattr(module, version_method)()
        else:
            version = getattr(module, '__version__', 'Unknown')
        print(" {}: {}".format(module_name, version))
        return True
    except ImportError as e:
        print(" {}: ERRO - {}".format(module_name, e))
        return False
    except Exception as e:
        print("  {}: {}".format(module_name, e))
        return False

def main():
    print("=" * 60)
    print(" TESTE DO AMBIENTE DEEPCAC - PYTHON 2.7")
    print("=" * 60)
    
    print("\n Python Version: {}".format(sys.version))
    print("Python Path: {}".format(sys.executable))
    
    print("\n TESTANDO DEPENDÊNCIAS:")
    print("-" * 40)
    
    # Lista de dependências para testar
    dependencies = [
        # Core scientific computing
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("matplotlib", "__version__"),
        
        # Deep Learning
        ("tensorflow", "__version__"),
        ("keras", "__version__"),
        
        # Medical imaging
        ("SimpleITK", "Version.VersionString"),
        ("skimage", "__version__"),
        
        # Data handling
        ("h5py", "__version__"),
        ("tables", "__version__"),
        
        # Configuration
        ("yaml", "__version__"),
        
        # Additional
        #("PIL", "__version__"),  # Pillow
        #("pandas", "__version__"),
        #("sklearn", "__version__"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, version_attr in dependencies:
        if test_import(module_name, version_attr):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(" RESULTADO: {}/{} dependências instaladas com sucesso".format(success_count, total_count))
    
    if success_count == total_count:
        print(" AMBIENTE CONFIGURADO COM SUCESSO!")
        print(" Todas as dependências estão funcionando corretamente")
        print("\n PRÓXIMOS PASSOS:")
        print("1. Seguir o guia de migração em MIGRATION_GUIDE.md")
        print("2. Atualizar o código para TensorFlow 2.x")
        print("3. Testar o pipeline completo")
    else:
        print("  ALGUMAS DEPENDÊNCIAS FALTANDO")
        print(" Verifique as instalações acima")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
