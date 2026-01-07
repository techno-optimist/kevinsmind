import { useState, useEffect } from 'react';

const BOOKS = [
  {
    id: 1,
    title: 'Gem',
    filename: 'Gem.pdf',
    description: 'A personal story of discovery and transformation.',
    cover: null
  },
  {
    id: 2,
    title: 'Parenting the Future',
    subtitle: 'Raising Resilient, Creative, and Ethical Humans in an AI-Driven World',
    filename: 'Parenting the Future_ Raising Resilient, Creative, and Ethical Humans in an AI-Driven World.pdf',
    description: 'A guide to raising children who can thrive alongside artificial intelligence, focusing on resilience, creativity, and ethics.',
    cover: null
  }
];

export default function Books() {
  const [selectedBook, setSelectedBook] = useState(null);

  const handleDownload = (book) => {
    const link = document.createElement('a');
    link.href = `/books/${encodeURIComponent(book.filename)}`;
    link.download = book.filename;
    link.click();
  };

  const handleRead = (book) => {
    window.open(`/books/${encodeURIComponent(book.filename)}`, '_blank');
  };

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Books</h2>
        <p>Written works by Kevin Russell. Click to read or download.</p>
      </div>

      <div className="tab-body">
        <div className="books-grid">
          {BOOKS.map(book => (
            <div
              key={book.id}
              className="book-card"
              onClick={() => setSelectedBook(book)}
            >
              <div className="book-cover">
                {book.cover ? (
                  <img src={book.cover} alt={book.title} />
                ) : (
                  <div className="book-cover-placeholder">
                    <BookIcon />
                  </div>
                )}
              </div>
              <div className="book-info">
                <h3 className="book-title">{book.title}</h3>
                {book.subtitle && (
                  <p className="book-subtitle">{book.subtitle}</p>
                )}
                <p className="book-description">{book.description}</p>
                <div className="book-actions">
                  <button
                    className="btn btn-primary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRead(book);
                    }}
                  >
                    <ReadIcon /> Read
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(book);
                    }}
                  >
                    <DownloadIcon /> Download
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Book Preview Modal */}
        {selectedBook && (
          <div className="book-modal-overlay" onClick={() => setSelectedBook(null)}>
            <div className="book-modal" onClick={(e) => e.stopPropagation()}>
              <button className="book-modal-close" onClick={() => setSelectedBook(null)}>
                <XIcon />
              </button>
              <div className="book-modal-content">
                <div className="book-modal-cover">
                  {selectedBook.cover ? (
                    <img src={selectedBook.cover} alt={selectedBook.title} />
                  ) : (
                    <div className="book-cover-placeholder large">
                      <BookIcon />
                    </div>
                  )}
                </div>
                <div className="book-modal-info">
                  <h2>{selectedBook.title}</h2>
                  {selectedBook.subtitle && (
                    <p className="book-subtitle">{selectedBook.subtitle}</p>
                  )}
                  <p className="book-description">{selectedBook.description}</p>
                  <div className="book-actions">
                    <button
                      className="btn btn-primary"
                      onClick={() => handleRead(selectedBook)}
                    >
                      <ReadIcon /> Read PDF
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => handleDownload(selectedBook)}
                    >
                      <DownloadIcon /> Download
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .books-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 1.5rem;
        }

        .book-card {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .book-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
          border-color: var(--accent);
        }

        .book-cover {
          height: 200px;
          background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          border-bottom: 1px solid var(--border);
        }

        .book-cover img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .book-cover-placeholder {
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-muted);
        }

        .book-cover-placeholder svg {
          width: 64px;
          height: 64px;
          opacity: 0.5;
        }

        .book-cover-placeholder.large svg {
          width: 120px;
          height: 120px;
        }

        .book-info {
          padding: 1.25rem;
        }

        .book-title {
          font-size: 1.25rem;
          font-weight: 600;
          margin: 0 0 0.5rem;
          color: var(--text-primary);
        }

        .book-subtitle {
          font-size: 0.875rem;
          color: var(--text-secondary);
          margin: 0 0 0.75rem;
          line-height: 1.4;
        }

        .book-description {
          font-size: 0.875rem;
          color: var(--text-muted);
          margin: 0 0 1rem;
          line-height: 1.5;
        }

        .book-actions {
          display: flex;
          gap: 0.75rem;
        }

        .book-actions .btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex: 1;
          justify-content: center;
        }

        .book-actions svg {
          width: 16px;
          height: 16px;
        }

        /* Modal Styles */
        .book-modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 2rem;
        }

        .book-modal {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: 16px;
          max-width: 600px;
          width: 100%;
          position: relative;
          overflow: hidden;
        }

        .book-modal-close {
          position: absolute;
          top: 1rem;
          right: 1rem;
          background: var(--bg-primary);
          border: 1px solid var(--border);
          border-radius: 8px;
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: var(--text-secondary);
          z-index: 10;
          transition: all 0.2s;
        }

        .book-modal-close:hover {
          background: var(--bg-secondary);
          color: var(--text-primary);
        }

        .book-modal-content {
          display: flex;
          flex-direction: column;
        }

        .book-modal-cover {
          height: 250px;
          background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          border-bottom: 1px solid var(--border);
        }

        .book-modal-cover img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .book-modal-info {
          padding: 1.5rem;
        }

        .book-modal-info h2 {
          font-size: 1.5rem;
          margin: 0 0 0.5rem;
        }

        .book-modal-info .book-subtitle {
          font-size: 1rem;
          margin-bottom: 1rem;
        }

        .book-modal-info .book-description {
          font-size: 1rem;
          margin-bottom: 1.5rem;
        }

        @media (max-width: 640px) {
          .books-grid {
            grid-template-columns: 1fr;
          }

          .book-modal {
            margin: 1rem;
            max-height: 90vh;
            overflow-y: auto;
          }
        }
      `}</style>
    </div>
  );
}

// === ICONS ===
function BookIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
    </svg>
  );
}

function ReadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  );
}
