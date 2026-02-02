if response.get('context_chunks'):
        with st.expander("View Sources"):
            for i, chunk in enumerate(response['context_chunks']):
                st.markdown(f"**Source {i+1}** (Page {chunk['page']}, {chunk['modality']}) - Score: {chunk['score']:.3f}")
                st.text(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                st.divider()
=======
    # Display sources
    if response.get('context_chunks'):
        with st.expander("View Sources"):
            for i, chunk in enumerate(response['context_chunks']):
                # Handle both SearchResult objects and dict formats
                if hasattr(chunk, 'chunk'):
                    # SearchResult object
                    page = chunk.chunk.page
                    modality = chunk.chunk.modality
                    score = chunk.score
                    content = chunk.chunk.content
                else:
                    # Dict format
                    page = chunk.get('page', 'N/A')
                    modality = chunk.get('modality', 'text')
                    score = chunk.get('score', 0.0)
                    content = chunk.get('content', '')

                st.markdown(f"**Source {i+1}** (Page {page}, {modality}) - Score: {score:.3f}")
                st.text(content[:500] + "..." if len(content) > 500 else content)
                st.divider()
