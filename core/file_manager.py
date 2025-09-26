import os
import struct
from typing import Union, List
from core.models import Table, Record

class FileManager:
    HEADER_SIZE = 4  
    
    def __init__(self, filename: str, table: Table):
        self.filename = filename
        self.header_filename = filename.replace('.dat', '.header')
        self.table = table
        self.record_size = table.record_size
        self.free_list_head = -1
        self.file_size = 0

        self._initialize_files()
    def _initialize_files(self):
        try: 
            with open(self.header_filename, 'rb') as f:
                data = f.read(self.HEADER_SIZE)
                if len(data) == self.HEADER_SIZE:
                    self.free_list_head = struct.unpack('i', data)[0]
                else:
                    self.free_list_head = -1 
        except FileNotFoundError:
            self.free_list_head = -1
            self._write_header()

        if os.path.exists(self.filename):
            file_bytes = os.path.getsize(self.filename)
            self.file_size = file_bytes // self.record_size
        else:
            self.file_size = 0

    def _write_header(self):
        try:
            with open(self.header_filename, 'wb') as f:
                f.write(struct.pack('i', self.free_list_head))
        except Exception as e:
            print(f"Error al escribir la cabecera: {e}")
            
    def _get_byte_offset(self, pos: int) -> int:
        return pos * self.record_size
    
    def add_record(self, record: Record) -> int:
        if self.free_list_head != -1:
            pos_to_use = self.free_list_head
            
            record_at_pos = self.read_record(pos_to_use)
            self.free_list_head = record_at_pos.next 
            
            record.next = 0 
            self._write_record_at_pos(record, pos_to_use)
            
            self._write_header() 
            
            return pos_to_use
        else:
            pos_to_use = self.file_size
            record.next = 0 
            self._write_record_at_pos(record, pos_to_use)
            
            self.file_size += 1 
            
            return pos_to_use
        
    def read_record(self, pos: int) -> Union[Record, None]:
            try:
                with open(self.filename, 'rb') as f:
                    offset = self._get_byte_offset(pos)
                    f.seek(offset)
                    data = f.read(self.record_size)
                    
                    if len(data) == self.record_size:
                        record = Record.unpack(self.table, data)
                        record.pos = pos 
                        return record
            except FileNotFoundError:
                print(f"Error: Archivo de datos '{self.filename}' no encontrado.")
            except Exception as e:
                print(f"Error al leer registro en pos {pos}: {e}")
                
            return None

    def _write_record_at_pos(self, record: Record, pos: int):
        with open(self.filename, 'r+b') as f:
            offset = self._get_byte_offset(pos)
            f.seek(offset)
            f.write(record.pack())
    
    def remove_record(self, pos: int) -> bool:
        record = self.read_record(pos)
        if record is None or record.next != 0:
            return False

        record.next = self.free_list_head
        
        self._write_record_at_pos(record, pos)
        
        self.free_list_head = pos
        
        self._write_header()
        
        return True