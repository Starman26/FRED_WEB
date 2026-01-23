#!/usr/bin/env python3
"""
diagnose_graph.py - Diagnóstico rápido del sistema multi-agente

Ejecutar: python scripts/diagnose_graph.py
"""
import os
import sys

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def diagnose():
    print("🔍 Diagnóstico Sistema Multi-Agente")
    print("=" * 50)
    
    # 1. Verificar variables de entorno
    print("\n1. VERIFICANDO VARIABLES DE ENTORNO...")
    
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL"]
    optional_vars = ["ANTHROPIC_API_KEY", "SUPABASE_SERVICE_ROLE_KEY", "DEFAULT_MODEL"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: Configurada ({value[:20]}...)")
        else:
            print(f"   ❌ {var}: FALTANTE")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: Configurada")
        else:
            print(f"   ⚠️  {var}: No configurada (opcional)")
    
    # 2. Verificar conexiones
    print("\n2. VERIFICANDO CONEXIONES...")
    
    # Supabase
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        if url and key:
            client = create_client(url, key)
            # Test query
            client.table("documents").select("id").limit(1).execute()
            print("   ✅ Supabase: Conectado")
        else:
            print("   ❌ Supabase: Faltan variables")
    except Exception as e:
        print(f"   ❌ Supabase: {e}")
    
    # Embeddings
    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        test_vec = emb.embed_query("test")
        print(f"   ✅ Embeddings: Dimensión {len(test_vec)}")
    except Exception as e:
        print(f"   ❌ Embeddings: {e}")
    
    # 3. Verificar imports
    print("\n3. VERIFICANDO IMPORTS...")
    
    imports_to_check = [
        ("src.agent.state", "AgentState"),
        ("src.agent.graph", "graph"),
        ("src.agent.contracts.worker_contract", "WorkerOutput"),
        ("src.agent.orchestrator", "orchestrator_plan_node"),
        ("src.agent.workers.research_node", "research_node"),
        ("src.agent.workers.tutor_node", "tutor_node"),
        ("src.agent.workers.troubleshooter_node", "troubleshooter_node"),
    ]
    
    for module_path, name in imports_to_check:
        try:
            module = __import__(module_path, fromlist=[name])
            obj = getattr(module, name)
            print(f"   ✅ {module_path}.{name}")
        except Exception as e:
            print(f"   ❌ {module_path}.{name}: {e}")
    
    # 4. Verificar grafo
    print("\n4. VERIFICANDO GRAFO...")
    
    try:
        from src.agent.graph import graph, get_graph_structure
        
        structure = get_graph_structure()
        print(f"   Nodos: {structure['nodes']}")
        print(f"   Workers: {structure['workers']}")
        print(f"   Orchestration: {structure['orchestration']}")
        print("   ✅ Grafo válido")
    except Exception as e:
        print(f"   ❌ Grafo: {e}")
    
    # 5. Test básico
    print("\n5. TEST BÁSICO...")
    
    try:
        from langchain_core.messages import HumanMessage
        from src.agent.graph import graph
        import uuid
        
        # IMPORTANTE: El grafo usa MemorySaver, necesita thread_id
        config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
        
        result = graph.invoke(
            {"messages": [HumanMessage(content="Hola, ¿qué puedes hacer?")]},
            config=config
        )
        
        if result.get("messages"):
            last_msg = result["messages"][-1]
            content = getattr(last_msg, "content", str(last_msg))
            print(f"   ✅ Respuesta: {content[:100]}...")
        else:
            print("   ⚠️  Sin respuesta en messages")
        
        # Verificar worker_outputs
        outputs = result.get("worker_outputs", [])
        print(f"   Worker outputs: {len(outputs)}")
        
    except Exception as e:
        print(f"   ❌ Test falló: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Diagnóstico completado.")


if __name__ == "__main__":
    diagnose()
