"""The store's whole view of an engine (docs/KV_SESSION_STORE_DESIGN.md §3.1). GemmaEngine conforms (Task 9);
tools/kvstore_store_test.mojo's FakeEngine conforms for CPU tests."""
from lib.kvstore.layout import KvLayout


trait KvSessionEngine:
    def kv_len(self) -> Int:
        ...

    def drafter_len(self) -> Int:
        ...

    def live_tokens_copy(self) -> List[Int]:
        ...

    def export_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, dst: UInt64) raises:
        """Copy tokens [a, b) of every component into host memory at dst, packed as the layout says.

        Copy contract: this must not return or raise while a copy into dst is still in flight. The store frees dst right
        after a raise (the chunk is not in its index yet, so nothing else would), and evicts or frees it as soon as the
        chunk is safely on disk. An engine that raises with an asynchronous copy still queued would have the device write
        into freed memory, so it must drain its copies before it raises."""
        ...

    def import_chunk(mut self, layout: KvLayout, a: Int, b: Int, drafter_end: Int, src: UInt64, src_tokens: Int, tokens: List[Int]) raises:
        """Copy the first b - a tokens of the chunk at src (laid out for src_tokens tokens) into the engine at [a, b).

        Copy contract: this must not return or raise while a copy out of src is still in flight. The store may evict or
        free src as soon as the call is over, a raise included. An engine that raises with an asynchronous copy still
        queued would read freed memory, so it must drain its copies before it raises."""
        ...

    def truncate_to(mut self, p: Int) raises:
        ...

    def reset_session(mut self) raises:
        ...
