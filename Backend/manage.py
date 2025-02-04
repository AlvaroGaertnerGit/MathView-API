#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def print_banner():
    banner_path = os.path.join(os.path.dirname(__file__), 'banner.txt')
    if os.path.exists(banner_path):
        with open(banner_path, 'r', encoding='utf-8') as banner_file:
            for line in banner_file:
                print(f"\033[1;36m{line.strip()}\033[0m")  # Cambia 1;32 por el color deseado

def main():

    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    print_banner()
    main()
