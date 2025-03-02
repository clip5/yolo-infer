#pragma once

namespace yoloinfer
{
    class Stream
    {
        public:
        Stream();
        ~Stream();
        void* get();
        int sync();
        private:
        void* stream_;
    };
    void stream_sync(void* stream);
}