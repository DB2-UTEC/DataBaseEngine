import requests
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import os

BASE_URL = "http://localhost:3001/api/query"
RESULTS_DIR = "benchmark_results"

# Crear directorio para resultados
os.makedirs(RESULTS_DIR, exist_ok=True)

class DatabaseBenchmark:
    def __init__(self):
        self.results = []
        
    def run_query(self, query, description=""):
        """Ejecutar query y retornar timing detallado"""
        try:
            start = time.perf_counter()
            response = requests.post(BASE_URL, json={"query": query}, timeout=30)
            end = time.perf_counter()
            
            if response.status_code != 200:
                return None
                
            result = response.json()
            timing = result.get('timing', {})
            
            return {
                'description': description,
                'query': query,
                'total_time_ms': (end - start) * 1000,
                'backend_time_ms': timing.get('total_ms', 0),
                'parse_time_ms': timing.get('parse_ms', 0),
                'exec_time_ms': timing.get('execution_ms', 0),
                'rows_returned': result.get('totalRows', 0),
                'success': True
            }
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_create_tables(self):
        """Crear tablas con diferentes índices"""
        print("📋 Creando tablas con diferentes índices...\n")
        
        csv_path = r"C:\Users\Sergio\BD2-Restructuracion\data\Restaurantes.csv"
        
        tables = [
            ("Restaurantes_ISAM", "ISAM"),
            ("Restaurantes_SEQ", "SEQUENTIAL"),
            ("Restaurantes_BTREE", "BTREE"),
            ("Restaurantes_HASH", "HASH"),
        ]
        
        for table_name, index_type in tables:
            query = f'CREATE TABLE {table_name} FROM FILE "{csv_path}" USING INDEX {index_type}("Restaurant_ID")'
            result = self.run_query(query, f"CREATE {index_type}")
            if result:
                print(f"✅ {table_name} creada ({result['backend_time_ms']:.2f}ms)")
            else:
                print(f"❌ Error creando {table_name}")
        
        print()
    
    def test_point_queries(self, iterations=10):
        """Prueba 1: Búsqueda por punto (WHERE id = x)"""
        print("🔍 Prueba 1: Búsqueda por Punto\n")
        
        test_ids = [100, 500, 1000, 5000, 10000]
        
        for test_id in test_ids:
            for index_type in ["ISAM", "SEQ", "BTREE", "HASH"]:
                times = []
                for _ in range(iterations):
                    query = f"SELECT * FROM Restaurantes_{index_type} WHERE Restaurant_ID = {test_id}"
                    result = self.run_query(query, f"Point Query - {index_type}")
                    if result:
                        times.append(result['exec_time_ms'])
                
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    self.results.append({
                        'test': 'point_query',
                        'index': index_type,
                        'parameter': test_id,
                        'avg_time_ms': avg_time,
                        'std_time_ms': std_time,
                        'min_time_ms': min(times),
                        'max_time_ms': max(times)
                    })
                    print(f"{index_type:10} ID={test_id:5} → {avg_time:6.2f}ms ±{std_time:.2f}ms")
        
        print()
    
    def test_range_queries(self, iterations=10):
        """Prueba 2: Búsqueda por rango (WHERE id BETWEEN x AND y)"""
        print("📊 Prueba 2: Búsqueda por Rango\n")
        
        ranges = [
            (1, 100),
            (1, 500),
            (1, 1000),
            (1, 5000),
            (1, 10000)
        ]
        
        for start, end in ranges:
            range_size = end - start
            for index_type in ["ISAM", "SEQ", "BTREE"]:
                times = []
                for _ in range(iterations):
                    query = f"SELECT * FROM Restaurantes_{index_type} WHERE Restaurant_ID BETWEEN {start} AND {end}"
                    result = self.run_query(query, f"Range Query - {index_type}")
                    if result:
                        times.append(result['exec_time_ms'])
                
                if times:
                    avg_time = np.mean(times)
                    self.results.append({
                        'test': 'range_query',
                        'index': index_type,
                        'parameter': range_size,
                        'avg_time_ms': avg_time,
                        'std_time_ms': np.std(times),
                        'min_time_ms': min(times),
                        'max_time_ms': max(times)
                    })
                    print(f"{index_type:10} Range size={range_size:5} → {avg_time:6.2f}ms")
        
        print()
    
    def test_insert_performance(self, iterations=5):
        """Prueba 3: Rendimiento de INSERT"""
        print("➕ Prueba 3: Rendimiento de INSERT\n")
        
        for index_type in ["ISAM", "SEQ", "BTREE", "HASH"]:
            times = []
            for i in range(iterations):
                new_id = 99000 + i
                query = f'INSERT INTO Restaurantes_{index_type} VALUES ({new_id}, "Test Restaurant", "Test City", "Test Cuisine", 4.5, 100, 50.0, "2025-01-01")'
                result = self.run_query(query, f"Insert - {index_type}")
                if result:
                    times.append(result['exec_time_ms'])
            
            if times:
                avg_time = np.mean(times)
                self.results.append({
                    'test': 'insert',
                    'index': index_type,
                    'parameter': iterations,
                    'avg_time_ms': avg_time,
                    'std_time_ms': np.std(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times)
                })
                print(f"{index_type:10} → {avg_time:6.2f}ms")
        
        print()
    
    def generate_graphs(self):
        """Generar todas las gráficas"""
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Gráfica 1: Comparación de búsqueda por punto
        self._plot_point_queries(df, timestamp)
        
        # Gráfica 2: Comparación de búsqueda por rango
        self._plot_range_queries(df, timestamp)
        
        # Gráfica 3: Comparación de INSERT
        self._plot_insert_performance(df, timestamp)
        
        # Gráfica 4: Tabla comparativa general
        self._plot_summary_table(df, timestamp)
        
        # Guardar resultados en JSON y CSV
        self._save_results(df, timestamp)
    
    def _plot_point_queries(self, df, timestamp):
        """Gráfica de búsquedas por punto"""
        point_data = df[df['test'] == 'point_query']
        
        if point_data.empty:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica de líneas
        for index in point_data['index'].unique():
            data = point_data[point_data['index'] == index]
            ax1.plot(data['parameter'], data['avg_time_ms'], marker='o', label=index, linewidth=2)
        
        ax1.set_xlabel('ID del Registro', fontsize=12)
        ax1.set_ylabel('Tiempo Promedio (ms)', fontsize=12)
        ax1.set_title('Rendimiento de Búsqueda por Punto', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Gráfica de barras (promedio general)
        avg_by_index = point_data.groupby('index')['avg_time_ms'].mean()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax2.bar(avg_by_index.index, avg_by_index.values, color=colors, alpha=0.8)
        ax2.set_ylabel('Tiempo Promedio (ms)', fontsize=12)
        ax2.set_title('Comparación General - Búsqueda por Punto', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/point_queries_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica guardada: point_queries_{timestamp}.png")
        plt.close()
    
    def _plot_range_queries(self, df, timestamp):
        """Gráfica de búsquedas por rango"""
        range_data = df[df['test'] == 'range_query']
        
        if range_data.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for index in range_data['index'].unique():
            data = range_data[range_data['index'] == index]
            ax.plot(data['parameter'], data['avg_time_ms'], marker='s', label=index, linewidth=2, markersize=8)
        
        ax.set_xlabel('Tamaño del Rango', fontsize=12)
        ax.set_ylabel('Tiempo Promedio (ms)', fontsize=12)
        ax.set_title('Rendimiento de Búsqueda por Rango', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/range_queries_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica guardada: range_queries_{timestamp}.png")
        plt.close()
    
    def _plot_insert_performance(self, df, timestamp):
        """Gráfica de rendimiento de INSERT"""
        insert_data = df[df['test'] == 'insert']
        
        if insert_data.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax.bar(insert_data['index'], insert_data['avg_time_ms'], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Tiempo Promedio (ms)', fontsize=12)
        ax.set_title('Rendimiento de INSERT por Tipo de Índice', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores y desviación estándar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            std = insert_data.iloc[i]['std_time_ms']
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms\n±{std:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/insert_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica guardada: insert_performance_{timestamp}.png")
        plt.close()
    
    def _plot_summary_table(self, df, timestamp):
        """Tabla resumen comparativa"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Crear resumen por tipo de índice
        summary_data = []
        for index in df['index'].unique():
            index_data = df[df['index'] == index]
            
            point_avg = index_data[index_data['test'] == 'point_query']['avg_time_ms'].mean()
            range_avg = index_data[index_data['test'] == 'range_query']['avg_time_ms'].mean()
            insert_avg = index_data[index_data['test'] == 'insert']['avg_time_ms'].mean()
            
            summary_data.append([
                index,
                f"{point_avg:.2f}" if not np.isnan(point_avg) else "N/A",
                f"{range_avg:.2f}" if not np.isnan(range_avg) else "N/A",
                f"{insert_avg:.2f}" if not np.isnan(insert_avg) else "N/A"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Índice', 'Búsqueda Punto (ms)', 'Búsqueda Rango (ms)', 'INSERT (ms)'],
                        cellLoc='center',
                        loc='center',
                        colColours=['#4ECDC4']*4)
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        plt.title('Resumen Comparativo de Rendimiento', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{RESULTS_DIR}/summary_table_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Tabla guardada: summary_table_{timestamp}.png")
        plt.close()
    
    def _save_results(self, df, timestamp):
        """Guardar resultados en JSON y CSV"""
        # JSON
        json_file = f'{RESULTS_DIR}/results_{timestamp}.json'
        df.to_json(json_file, orient='records', indent=2)
        print(f"✅ Resultados JSON: results_{timestamp}.json")
        
        # CSV
        csv_file = f'{RESULTS_DIR}/results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✅ Resultados CSV: results_{timestamp}.csv")


def main():
    print("=" * 60)
    print("  SISTEMA DE BENCHMARKING - BASE DE DATOS")
    print("=" * 60)
    print()
    
    benchmark = DatabaseBenchmark()
    
    # Ejecutar pruebas
    benchmark.test_create_tables()
    benchmark.test_point_queries(iterations=10)
    benchmark.test_range_queries(iterations=10)
    benchmark.test_insert_performance(iterations=5)
    
    # Generar gráficas
    print("\n📊 Generando gráficas...")
    benchmark.generate_graphs()
    
    print("\n✅ Benchmark completado!")
    print(f"📁 Resultados guardados en: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()