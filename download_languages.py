from huggingface_hub import snapshot_download
from fasttext_languages import languages
import asyncio


async def download_lang(lang, semaphore):
    try:
        async with semaphore:
            await asyncio.to_thread(
                snapshot_download,
                repo_id="MaLA-LM/mala-monolingual-dedup",
                repo_type="dataset",
                allow_patterns=f"{lang}*/data-00000-of-*.arrow",
                local_dir="",  # path to local directory
            )
            print(f"Downloaded for language: {lang}")
    except Exception as error:
        print(f"Error downloading {lang}: {error}")


async def main():
    semaphore = asyncio.Semaphore(5)
    async with asyncio.TaskGroup() as tg:
        for lang in languages:
            print(f"Starting download for {lang}...")
            tg.create_task(download_lang(lang, semaphore))


asyncio.run(main())
