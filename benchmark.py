import os
import time
import faiss
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Configuração de conexão (usa porta 5433 por padrão para evitar conflito com Postgres local na 5432)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "senha123")


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 80)
    log("TRABALHO DE BANCO DE DADOS: Índices Vetoriais e Busca Semântica para IA")
    log("Comparativo Prático: PostgreSQL (ILIKE) vs FAISS + Sentence Transformers")
    log("=" * 80)

    # 1. Configurações Iniciais
    log("\n[1/5] Carregando modelo de IA 'all-MiniLM-L6-v2' (384 dimensões)...")
    t0 = time.perf_counter()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    log(f"      Modelo carregado em {(time.perf_counter() - t0):.2f}s.")

    # Gera base de documentos simulados + documentos-alvo reais
    num_records = 50000
    log(f"\n[2/5] Gerando base de {num_records} documentos corporativos...")

    templates_ruido = [
        "documento de teste numero {i} sobre administracao de banco de dados e infraestrutura",
        "ata de reuniao tecnica numero {i} sobre desenvolvimento de software e microsservicos",
        "registro operacional numero {i} referente a manutencao de redes e servidores linux",
        "chamado de suporte interno numero {i} para atualizacao de pacotes e seguranca",
        "manual de procedimentos internos numero {i} sobre politicas de backup e armazenamento",
    ]
    textos = [templates_ruido[i % len(templates_ruido)].format(i=i) for i in range(num_records)]

    # Documentos-alvo que queremos encontrar na busca
    doc_alvo_financeiro = "relatorio financeiro de lucros, receitas e perdas contabeis do ultimo trimestre"
    doc_alvo_ia = "arquitetura de busca vetorial e inteligencia artificial generativa para modelos LLM"
    doc_alvo_sec = "auditoria de ciberseguranca sobre ataque ransomware, phishing e criptografia de dados"

    textos.extend([doc_alvo_financeiro, doc_alvo_ia, doc_alvo_sec])
    total_docs = len(textos)

    # 2. Setup do PostgreSQL (Docker)
    log(f"\n[3/5] Configurando PostgreSQL ({DB_HOST}:{DB_PORT}) e inserindo {total_docs} registros...")
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documentos;")
    cur.execute("CREATE TABLE documentos (id SERIAL PRIMARY KEY, conteudo TEXT);")

    t_insert = time.perf_counter()
    execute_values(
        cur,
        "INSERT INTO documentos (conteudo) VALUES %s",
        [(txt,) for txt in textos],
        page_size=5000,
    )
    conn.commit()
    log(f"      Inserção concluída em {(time.perf_counter() - t_insert):.2f}s.")

    # 3. Setup do FAISS (Busca Vetorial)
    log("\n[4/5] Gerando embeddings da base e construindo índices vetoriais FAISS...")
    t_emb = time.perf_counter()
    # Codifica lote de frases base reais + alvos
    amostra_textos = textos[:200]
    emb_amostra = model.encode(amostra_textos, normalize_embeddings=True).astype(np.float32)
    emb_alvos = model.encode(
        [doc_alvo_financeiro, doc_alvo_ia, doc_alvo_sec], normalize_embeddings=True
    ).astype(np.float32)

    # Replica os embeddings reais dos documentos de ruído para compor os 50.000 vetores no índice
    repeticoes = (num_records // len(emb_amostra)) + 1
    matriz_base = np.tile(emb_amostra, (repeticoes, 1))[:num_records]
    embeddings = np.vstack([matriz_base, emb_alvos]).astype(np.float32)

    dimensao = embeddings.shape[1]

    # Índice 1: Flat L2 (Busca Exata por Força Bruta Otimizada em C++/SIMD)
    indice_flat = faiss.IndexFlatL2(dimensao)
    indice_flat.add(embeddings)

    # Índice 2: HNSW (Hierarchical Navigable Small World - Grafo aproximado usado em produção RAG)
    indice_hnsw = faiss.IndexHNSWFlat(dimensao, 32)
    indice_hnsw.hnsw.efSearch = 64
    indice_hnsw.add(embeddings)

    log(f"      Índices FAISS (FlatL2 e HNSW) prontos em {(time.perf_counter() - t_emb):.2f}s (Dimensão={dimensao}).")

    # 4. Experimento 1: Eficácia Semântica vs Palavra-Chave Exata
    log("\n" + "=" * 80)
    log("PARTE A: EFICÁCIA DA BUSCA (PALAVRA-CHAVE EXATA vs CONTEXTO SEMÂNTICO)")
    log("=" * 80)

    casos_teste = [
        {
            "cenario": "Cenário 1: Palavra-chave literal presente ('lucro')",
            "pergunta": "cadê o balanço de lucros?",
            "sql_where": "conteudo ILIKE '%lucro%'",
            "alvo_esperado": doc_alvo_financeiro,
        },
        {
            "cenario": "Cenário 2: Busca por sinônimo/contexto financeiro (SEM palavra exata no filtro)",
            "pergunta": "qual foi o balanço monetário e faturamento trimestral da empresa?",
            "sql_where": "conteudo ILIKE '%monetario%' OR conteudo ILIKE '%faturamento%'",
            "alvo_esperado": doc_alvo_financeiro,
        },
        {
            "cenario": "Cenário 3: Busca conceitual técnica (RAG / IA)",
            "pergunta": "como usar embeddings e vetores em modelos de inteligência artificial como ChatGPT?",
            "sql_where": "conteudo ILIKE '%embeddings%' OR conteudo ILIKE '%ChatGPT%'",
            "alvo_esperado": doc_alvo_ia,
        },
        {
            "cenario": "Cenário 4: Busca por incidente de segurança (Zero palavras em comum)",
            "pergunta": "alerta de virus malicioso, fraude por email e sequestro de arquivos",
            "sql_where": "conteudo ILIKE '%virus%' OR conteudo ILIKE '%fraude%' OR conteudo ILIKE '%sequestro%'",
            "alvo_esperado": doc_alvo_sec,
        },
    ]

    acertos_pg = 0
    acertos_faiss = 0

    for caso in casos_teste:
        log(f"\n>>> {caso['cenario']}")
        log(f"    Pergunta do usuário : '{caso['pergunta']}'")
        log(f"    Filtro SQL usado    : WHERE {caso['sql_where']}")

        # PostgreSQL
        cur.execute(f"SELECT conteudo FROM documentos WHERE {caso['sql_where']} LIMIT 3;")
        res_pg = [r[0] for r in cur.fetchall()]
        pg_achou = caso["alvo_esperado"] in res_pg
        if pg_achou:
            acertos_pg += 1

        # FAISS
        vetor_q = model.encode([caso["pergunta"]], normalize_embeddings=True).astype(np.float32)
        dist, idx = indice_flat.search(vetor_q, 3)
        res_faiss = [textos[i] for i in idx[0]]
        faiss_achou = caso["alvo_esperado"] in res_faiss
        if faiss_achou:
            acertos_faiss += 1

        log(
            f"    [PostgreSQL ILIKE]  -> {'ENCONTROU' if pg_achou else 'FALHOU (0 resultados)'} | Top-1: {res_pg[0] if res_pg else 'Nenhum'}"
        )
        log(
            f"    [FAISS Semântico]   -> {'ENCONTROU' if faiss_achou else 'FALHOU'} (Dist L2={dist[0][0]:.4f}) | Top-1: {res_faiss[0]}"
        )

    taxa_pg = (acertos_pg / len(casos_teste)) * 100
    taxa_faiss = (acertos_faiss / len(casos_teste)) * 100
    log(f"\nResumo de Recall@3 nos cenários: PostgreSQL = {taxa_pg:.0f}% ({acertos_pg}/4) | FAISS = {taxa_faiss:.0f}% ({acertos_faiss}/4)")

    # 5. Experimento 2: Benchmark Numérico de Latência (Média de 50 execuções)
    log("\n" + "=" * 80)
    log("PARTE B: MEDIDAS NUMÉRICAS DE LATÊNCIA (50.003 REGISTROS, MÉDIA DE 50 EXECUÇÕES)")
    log("=" * 80)

    pergunta_bench = "cadê o balanço de lucros?"
    vetor_pergunta = model.encode([pergunta_bench], normalize_embeddings=True).astype(np.float32)
    n_runs = 50

    # 5.1 Tempo PostgreSQL ILIKE (1 termo vs múltiplos termos como em buscas reais)
    tempos_pg_simples = []
    tempos_pg_multi = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cur.execute("SELECT conteudo FROM documentos WHERE conteudo ILIKE '%lucro%' LIMIT 5;")
        _ = cur.fetchall()
        tempos_pg_simples.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cur.execute(
            "SELECT conteudo FROM documentos WHERE conteudo ILIKE '%lucro%' OR conteudo ILIKE '%balanco%' OR conteudo ILIKE '%financeiro%' LIMIT 5;"
        )
        _ = cur.fetchall()
        tempos_pg_multi.append((time.perf_counter() - t0) * 1000)

    # 5.2 Tempo FAISS (Apenas busca no índice vs Pipeline completo com Embedding)
    tempos_faiss_flat = []
    tempos_faiss_hnsw = []
    tempos_embedding = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        vetor_q = model.encode([pergunta_bench], normalize_embeddings=True).astype(np.float32)
        tempos_embedding.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        indice_flat.search(vetor_q, 5)
        tempos_faiss_flat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        indice_hnsw.search(vetor_q, 5)
        tempos_faiss_hnsw.append((time.perf_counter() - t0) * 1000)

    media_pg_simples = float(np.mean(tempos_pg_simples))
    media_pg_multi = float(np.mean(tempos_pg_multi))
    media_faiss_flat = float(np.mean(tempos_faiss_flat))
    media_faiss_hnsw = float(np.mean(tempos_faiss_hnsw))
    media_embedding = float(np.mean(tempos_embedding))
    media_pipeline_total = media_embedding + media_faiss_hnsw

    speedup_flat = media_pg_simples / media_faiss_flat
    speedup_hnsw = media_pg_simples / media_faiss_hnsw

    log(f"1. PostgreSQL (ILIKE 1 termo - Seq Scan)     : {media_pg_simples:.3f} ms (±{np.std(tempos_pg_simples):.3f} ms)")
    log(f"2. PostgreSQL (ILIKE 3 termos - Seq Scan)    : {media_pg_multi:.3f} ms (±{np.std(tempos_pg_multi):.3f} ms)")
    log(f"3. FAISS IndexFlatL2 (Busca Vetorial Exata)  : {media_faiss_flat:.3f} ms (±{np.std(tempos_faiss_flat):.3f} ms) -> {speedup_flat:.1f}x mais rápido que SQL!")
    log(f"4. FAISS IndexHNSW (Busca em Grafo ANN)      : {media_faiss_hnsw:.3f} ms (±{np.std(tempos_faiss_hnsw):.3f} ms) -> {speedup_hnsw:.1f}x mais rápido que SQL!")
    log(f"5. Geração do Embedding da Query (CPU)       : {media_embedding:.3f} ms (±{np.std(tempos_embedding):.3f} ms)")
    log(f"6. Pipeline RAG Completo (Embed + FAISS HNSW): {media_pipeline_total:.3f} ms")

    # 6. Experimento 3: Curva de Escalabilidade (5k, 10k, 25k, 50k registros)
    log("\n" + "=" * 80)
    log("PARTE C: ESCALABILIDADE COM O AUMENTO DO VOLUME DE DADOS (N = 5k a 50k)")
    log("=" * 80)

    tamanhos = [5000, 10000, 25000, 50000]
    escala_rows = []

    for n in tamanhos:
        # Sub-índice FAISS Flat e HNSW com n elementos
        sub_emb = embeddings[:n]
        sub_flat = faiss.IndexFlatL2(dimensao)
        sub_flat.add(sub_emb)

        sub_hnsw = faiss.IndexHNSWFlat(dimensao, 32)
        sub_hnsw.hnsw.efSearch = 32
        sub_hnsw.add(sub_emb)

        # Mede SQL varrendo os primeiros n IDs
        t_sql_list = []
        t_flat_list = []
        t_hnsw_list = []
        for _ in range(25):
            t0 = time.perf_counter()
            cur.execute(
                "SELECT conteudo FROM documentos WHERE id <= %s AND conteudo ILIKE '%%lucro%%';",
                (n,),
            )
            _ = cur.fetchall()
            t_sql_list.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            sub_flat.search(vetor_pergunta, 5)
            t_flat_list.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            sub_hnsw.search(vetor_pergunta, 5)
            t_hnsw_list.append((time.perf_counter() - t0) * 1000)

        m_sql = float(np.mean(t_sql_list))
        m_flat = float(np.mean(t_flat_list))
        m_hnsw = float(np.mean(t_hnsw_list))
        escala_rows.append(
            {
                "num_registros": n,
                "postgresql_ilike_ms": round(m_sql, 4),
                "faiss_flat_ms": round(m_flat, 4),
                "faiss_hnsw_ms": round(m_hnsw, 4),
                "speedup_hnsw_vs_sql": round(m_sql / m_hnsw, 1),
            }
        )
        log(
            f"   N = {n:6d} docs | PostgreSQL ILIKE: {m_sql:7.3f} ms | FAISS FlatL2: {m_flat:6.3f} ms | FAISS HNSW: {m_hnsw:6.3f} ms ({m_sql/m_hnsw:.0f}x speedup)"
        )

    df_escala = pd.DataFrame(escala_rows)
    df_escala.to_csv("metricas_benchmark.csv", index=False)

    # 7. Geração dos Gráficos para a Apresentação
    log("\n[5/5] Gerando gráficos de alta resolução para os slides...")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)

    # Subgráfico 1: Latência de Busca no Banco/Índice (50k docs)
    labels_lat = ["PostgreSQL\n(ILIKE 3 termos)", "PostgreSQL\n(ILIKE 1 termo)", "FAISS\n(IndexFlatL2)", "FAISS\n(IndexHNSW)"]
    valores_lat = [media_pg_multi, media_pg_simples, media_faiss_flat, media_faiss_hnsw]
    cores_lat = ["#c0392b", "#2980b9", "#27ae60", "#16a085"]

    bars = axes[0].bar(labels_lat, valores_lat, color=cores_lat, width=0.55, edgecolor="black", linewidth=0.6)
    axes[0].set_ylabel("Tempo de Resposta (ms) - Menor é melhor", fontweight="bold")
    axes[0].set_title("Latência de Busca no Índice (50.000 docs)", fontweight="bold", pad=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, valores_lat):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(valores_lat) * 0.02,
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(valores_lat) * 1.18)

    # Subgráfico 2: Curva de Escalabilidade (N vs Tempo)
    axes[1].plot(df_escala["num_registros"], df_escala["postgresql_ilike_ms"], marker="o", linewidth=2.5, color="#2980b9", label="PostgreSQL (ILIKE Seq Scan)")
    axes[1].plot(df_escala["num_registros"], df_escala["faiss_flat_ms"], marker="s", linewidth=2.5, color="#27ae60", label="FAISS IndexFlatL2 (Exato)")
    axes[1].plot(df_escala["num_registros"], df_escala["faiss_hnsw_ms"], marker="^", linewidth=2.5, color="#16a085", label="FAISS IndexHNSW (Grafo ANN)")
    axes[1].set_xlabel("Volume de Documentos (N)", fontweight="bold")
    axes[1].set_ylabel("Latência de Busca (ms)", fontweight="bold")
    axes[1].set_title("Escalabilidade: Crescimento Linear vs Vetorial", fontweight="bold", pad=12)
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subgráfico 3: Taxa de Sucesso (Recall@3) em Buscas Contextuais/Sinônimos
    labels_acc = ["PostgreSQL\n(Palavra-Chave ILIKE)", "FAISS + Embeddings\n(Busca Semântica)"]
    valores_acc = [taxa_pg, taxa_faiss]
    bars_acc = axes[2].bar(labels_acc, valores_acc, color=["#7f8c8d", "#2ecc71"], width=0.5, edgecolor="black", linewidth=0.6)
    axes[2].set_ylabel("Taxa de Acerto / Recall@3 (%)", fontweight="bold")
    axes[2].set_title("Eficácia em Buscas por Contexto e Sinônimos", fontweight="bold", pad=12)
    axes[2].set_ylim(0, 115)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars_acc, valores_acc):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.suptitle(
        "Benchmark Experimental: Busca Textual Tradicional (SQL) vs Busca Vetorial Semântica (FAISS)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("grafico_comparativo.png", bbox_inches="tight", dpi=150)
    log("      -> Gráfico salvo em: grafico_comparativo.png")
    log("      -> Tabela CSV salva em: metricas_benchmark.csv")

    with open("resultado_benchmark.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log("      -> Log completo salvo em: resultado_benchmark.txt")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
