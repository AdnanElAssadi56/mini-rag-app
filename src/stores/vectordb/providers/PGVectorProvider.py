from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import (
    DistanceMethodEnums,
    PgVectorTableSchemaEnums,
    PgVectorDistanceMethodEnums,
    PgVectorIndexTypeEnums
)
import logging
from typing import List
from models.db_schemas import RetrievedDocument 
from sqlalchemy.sql import text as sql_text
import json

class PGVectorProvider(VectorDBInterface):
    def __init__(self, db_client, default_vector_size: int = 786,
                 distance_method: str= None, index_threshold: int=100):
                 
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold

        self.pgvector_table_prefix = PgVectorTableSchemaEnums._PREFIX.value

        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        
        if distance_method == DistanceMethodEnums.DOT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value

        self.distance_method = distance_method
        self.logger = logging.getLogger("uvicorn.error")
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"
        
    async def connect(self):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(
                    sql_text(
                        f"""
                        CREATE EXTENSION IF NOT EXISTS vector;
                        """
                    )
                )
                await session.commit()

    async def disconnect(self):
        pass

    async def is_collection_existed(self, collection_name: str) -> bool:
        async with self.db_client() as session:
            async with session.begin():
                results = await session.execute(
                    sql_text(
                        f"""
                        SELECT * FROM pg_tables WHERE tablename = :collection_name
                        """
                    ),
                    {"collection_name": collection_name}
                )
                record = results.scalar_one_or_none()
                return record is not None
            
    async def list_all_collections(self) -> List[str]:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                results = await session.execute(
                    sql_text(
                        f"""
                        SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix;
                        """
                    ),
                    {"prefix": f"{self.pgvector_table_prefix}%"}
                )
                records = results.scalars().all()
        
        return records
            
    async def get_collection_info(self, collection_name: str) -> dict:  
        async with self.db_client() as session:
            async with session.begin():
                table_info_stmt = sql_text(
                    f"""
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes
                    FROM pg_tables
                    WHERE tablename = :collection_name
                    """
                )
                count_sql = sql_text(
                    f"""
                    SELECT COUNT(*) FROM {collection_name};
                    """
                )
                table_info = await session.execute(
                    table_info_stmt,
                    {"collection_name": collection_name}
                )
                record_counts = await session.execute(count_sql)
                table_data = table_info.fetchone()
                if not table_data:
                    return None
                
                return {
                    "table_info": dict(table_data._mapping),
                    "record_counts": record_counts.scalar_one()
                }
            
    async def delete_collection(self, collection_name: str):
        async with self.db_client() as session:
            self.logger.info(f"Deleting collection: {collection_name}")
            async with session.begin():
                await session.execute(
                    sql_text(f"DROP TABLE IF EXISTS {collection_name};")
                )
                await session.commit()
        
        return True

    async def create_collection(self, collection_name: str,
                                      embedding_size: int,
                                      do_reset: bool = False):
        if do_reset:
            await self.delete_collection(collection_name)
        
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.info(f"Creating Collection: {collection_name}")
            async with self.db_client() as session:
                async with session.begin():
                    await session.execute(sql_text(
                            f'CREATE TABLE {collection_name} ('
                                f'{PgVectorTableSchemaEnums.ID.value} bigserial PRIMARY KEY, '
                                f'{PgVectorTableSchemaEnums.TEXT.value} text, '
                                f'{PgVectorTableSchemaEnums.VECTOR.value} vector({embedding_size}), '
                                f'{PgVectorTableSchemaEnums.METADATA.value} jsonb DEFAULT \'{{}}\', '
                                f'{PgVectorTableSchemaEnums.CHUNK_ID.value} integer, '
                                f'FOREIGN KEY ({PgVectorTableSchemaEnums.CHUNK_ID.value}) REFERENCES chunks(chunk_id)'
                            ')'
                        ))
                    await session.commit()

            return True
        
        return False
    
    async def is_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text("""
                                     SELECT 1
                                     FROM pg_indexes
                                     WHERE tablename = :collection_name
                                     AND indexname = :index_name
                                     """)
                results = await session.execute(check_sql,{
                                                    "index_name": index_name,
                                                    "collection_name": collection_name
                                                })
                return bool(results.scalar_one_or_none())
            
    async def create_vector_index(self, collection_name: str, 
                                        index_type: str = PgVectorIndexTypeEnums.HNSW.value):
        is_index_existed = await self.is_index_existed(collection_name)
        if is_index_existed:
            return False

        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if records_count < self.index_threshold:
                    return False
                
                self.logger.info(f"START: Creating vector index for collection: {collection_name}")

                index_name = self.default_index_name(collection_name)
                vector_column = PgVectorTableSchemaEnums.VECTOR.value

                create_index_sql = sql_text(
                                            f'CREATE INDEX {index_name} ON {collection_name} '
                                            f'USING {index_type} ({vector_column} {self.distance_method})'
                                            )
                await session.execute(create_index_sql)

                self.logger.info(f"END: Created vector index for collection: {collection_name}")

    async def reset_index(self, collection_name: str, index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> bool:

        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                drop_sql = sql_text(f"DROP INDEX IF EXISTS {index_name}")
                await session.execute(drop_sql)

        return await self.create_vector_index(collection_name=collection_name, index_type=index_type)
        

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        if not record_id:
            self.logger.error(f"Can not insert new record without chunk_id: {collection_name}")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(f"""
                    INSERT INTO {collection_name}
                        ({PgVectorTableSchemaEnums.TEXT.value},
                        {PgVectorTableSchemaEnums.VECTOR.value},
                        {PgVectorTableSchemaEnums.METADATA.value},
                        {PgVectorTableSchemaEnums.CHUNK_ID.value})
                    VALUES
                        (:text, :vector, :metadata, :chunk_id)
                """)
                await session.execute(
                    insert_sql,
                    {
                        "text": text,
                        "vector": "[" + ",".join([ str(v) for v in vector ]) + "]",
                        "metadata": json.dumps(metadata) if metadata is not None else "{}",
                        "chunk_id": record_id
                    }
                )
                await session.commit()

        await self.create_vector_index(collection_name=collection_name)
        return True
    
    async def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        if len(vectors) != len(record_ids):
            self.logger.error(f"Invalid data items for collection: {collection_name}")
            return False
        
        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)
        
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_vectors = vectors[i:i+batch_size]
                    batch_metadata = metadata[i:i+batch_size]
                    batch_record_ids = record_ids[i:i+batch_size]

                    values = []

                    for _text, _vector, _metadata, _record_id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids):
                        values.append({
                            'text': _text,
                            'vector': "[" + ",".join([ str(v) for v in _vector ]) + "]",
                            'metadata': json.dumps(_metadata) if _metadata is not None else "{}",
                            'chunk_id': _record_id
                        })

                    batch_insert_sql = sql_text(f'INSERT INTO {collection_name} '
                                    f'({PgVectorTableSchemaEnums.TEXT.value}, '
                                    f'{PgVectorTableSchemaEnums.VECTOR.value}, '
                                    f'{PgVectorTableSchemaEnums.METADATA.value}, '
                                    f'{PgVectorTableSchemaEnums.CHUNK_ID.value}) '
                                    f'VALUES (:text, :vector, :metadata, :chunk_id)')
                    
                    await session.execute(batch_insert_sql, values)
        await self.create_vector_index(collection_name=collection_name)
        return True
                
    async def search_by_vector(self, collection_name: str, vector: list, limit: int):

        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        vector_column = PgVectorTableSchemaEnums.VECTOR.value
        text_column = PgVectorTableSchemaEnums.TEXT.value
        metadata_column = PgVectorTableSchemaEnums.METADATA.value
        chunk_id_column = PgVectorTableSchemaEnums.CHUNK_ID.value

        vector = "[" + ",".join([ str(v) for v in vector ]) + "]"
        async with self.db_client() as session:
            async with session.begin():
                search_sql = sql_text(f'SELECT {text_column} AS text, '
                                    f' 1 - ({vector_column} <=> :query_vector) AS score'
                                    f' FROM {collection_name}'
                                    f' ORDER BY score DESC'
                                    f' LIMIT {limit}'
                                    )
                results = await session.execute(search_sql, {"query_vector": vector,})
                records = results.fetchall()

                return [
                    RetrievedDocument(
                        text = record.text,
                        score = record.score
                    ) for record in records
                ]
            